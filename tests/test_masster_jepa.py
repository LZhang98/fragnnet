"""Step 1 of the MASSter JEPA spectrum-embedding auxiliary.

``docs/plans/masster/masster_jepa_embedding_plan.md`` adds one default-off, training-only loss
term: a small MLP predicts a frozen embedding of each spectrum's measured spectrum from a pooled
representation of the model, and ``weight * (1 - cosine)`` joins the loss. At weight 0 nothing
exists (no module, no table, no key), so a run is exactly ``i1``. Spectra without a target add
exactly 0. The predictor is applied inside ``loss_fn``, so inference never builds its output.
"""

from pathlib import Path

import lightning as L
import numpy as np
import pytest
import torch as th
import torch.nn.functional as F
from rdkit import Chem
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

import fragnnet.runner as runner
from fragnnet.model.bond_fragment_heads import _segment_mean
from fragnnet.pl_model import MassterPL
from fragnnet.utils.feat_utils import get_mol_graph
from fragnnet.utils.jepa_utils import (
    JEPA_EMBEDDING_DIM,
    JepaTargetTable,
    build_jepa_predictor,
    load_jepa_target_table,
    masked_cosine_distance,
    raw_peak_signature,
    write_jepa_table,
)
from tests.test_masster_memorization_route import _forward, _head, _head_inputs
from tests.test_masster_model import _build_batch, _build_model, _parameter_gradient_sum
from tests.test_masster_pl import _build_pl, _build_selection_only_pl, _escape_inputs

JEPA_PREFIX = "jepa_predictor."
JEPA_NAMES = {"jepa_loss", "jepa_cos", "jepa_target_fraction"}
# Spectra 5 and 9 are in the table without a target (fewer than 3 peaks after the floor).
TABLE_IDS = (3, 5, 7, 9, 11, 20)
NO_TARGET_IDS = (5, 9)


@pytest.fixture(autouse=True)
def _fresh_table_cache():
    load_jepa_target_table.cache_clear()
    yield
    load_jepa_target_table.cache_clear()


def _peaks(spec_id: int) -> np.ndarray:
    """Raw m/z of one spectrum: 3 + id % 5 distinct peaks, different for every id."""
    return 100.0 + 0.37 * spec_id + 13.1 * np.arange(3 + spec_id % 5)


def _write_table(
    path: Path, ids: tuple[int, ...] = TABLE_IDS, no_target: tuple[int, ...] = NO_TARGET_IDS
) -> str:
    has_target = np.array([spec_id not in no_target for spec_id in ids])
    emb = np.random.default_rng(0).normal(size=(len(ids), JEPA_EMBEDDING_DIM)).astype(np.float16)
    emb[~has_target] = 0
    signatures = np.array([raw_peak_signature(_peaks(spec_id)) for spec_id in ids])
    write_jepa_table(
        str(path),
        spec_id=np.array(ids),
        has_target=has_target,
        emb=emb,
        n_peaks=signatures[:, 0],
        mz_sum=signatures[:, 1],
        source="unit test",
    )
    return str(path)


def _peak_batch(ids: list[int], peaks: list[np.ndarray] | None = None) -> dict[str, th.Tensor]:
    """The three batch keys the hook reads, as the dataset collates them (float32 m/z)."""
    peaks = [_peaks(spec_id) for spec_id in ids] if peaks is None else peaks
    return {
        "spec_unique_id": th.tensor(ids),
        "spec_mzs": th.cat([th.tensor(mzs.tolist(), dtype=th.float) for mzs in peaks]),
        "spec_batch_idxs": th.repeat_interleave(
            th.arange(len(ids)), th.tensor([len(mzs) for mzs in peaks])
        ),
    }


def _lookup(table: JepaTargetTable, batch: dict[str, th.Tensor]):
    return table.lookup(batch["spec_unique_id"], batch["spec_mzs"], batch["spec_batch_idxs"])


@pytest.fixture
def table_fp(tmp_path: Path) -> str:
    return _write_table(tmp_path / "table.npz")


def _i1_like() -> dict:
    """What the template lacks of i1: unmerged spectra and the vectorized sparse loss."""
    config = runner.load_config(
        str(Path(__file__).parents[1] / "config"), None, None, model_type="masster"
    )
    return {
        "spec_params": {**config["spec_params"], "merge": False},
        "loss_vectorized": True,
        "masster_unique_fragment_target": 4,
        "masster_ranked_mask_budget": 8,
    }


def _selection_pl(**overrides):
    """A proposal-selection-only PL module, i1's mode, with the term off unless asked."""
    return _build_selection_only_pl(**{**_i1_like(), **overrides})[0]


def _jepa_pl(source: str = "encoder", weight: float = 0.5, fp: str = "unused.npz", **overrides):
    return _selection_pl(
        masster_jepa_loss_weight=weight,
        masster_jepa_source=source,
        masster_jepa_target_fp=fp,
        **overrides,
    )


def _seeded(builder, **kwargs):
    th.manual_seed(17)
    return builder(**kwargs)


def _loss_inputs(output: dict) -> dict:
    """One true peak per spectrum, on a predicted m/z, as the channel-support test does."""
    mzs, batch_idxs = output["pred_mzs"], output["pred_batch_idxs"]
    num_spectra = int(batch_idxs.max()) + 1
    inputs = dict(output)
    inputs.update(
        true_mzs=th.stack([mzs[batch_idxs == index][0] for index in range(num_spectra)]).float(),
        true_intensities=th.ones(num_spectra),
        true_logprobs=th.zeros(num_spectra),
        true_batch_idxs=th.arange(num_spectra),
        batch_size=num_spectra,
    )
    return inputs


def _hooked(pl, ids: list[int]) -> dict:
    """The two-spectrum escape batch plus ids and peaks, after the Lightning hook."""
    return pl.on_before_batch_transfer({**_escape_inputs(), **_peak_batch(ids)}, 0)


# 1. The side table.


def test_lookup_returns_targets_zero_rows_for_masked_spectra_and_a_copy(table_fp: str) -> None:
    table = JepaTargetTable(table_fp)
    stored = table.embeddings.copy()
    row = {int(spec_id): index for index, spec_id in enumerate(table.spec_ids)}

    target, valid = _lookup(table, _peak_batch([7, 5, 3]))

    assert valid.tolist() == [True, False, True] and valid.dtype == th.bool
    assert target.dtype == th.float16 and target.shape == (3, JEPA_EMBEDDING_DIM)
    assert th.equal(target[0], th.from_numpy(stored[row[7]]))
    assert th.equal(target[2], th.from_numpy(stored[row[3]]))
    assert not target[1].any()
    target.zero_()
    assert np.array_equal(table.embeddings, stored)


@pytest.mark.parametrize("unknown", [4, 10**9])
def test_lookup_rejects_a_spectrum_the_table_was_not_built_for(table_fp: str, unknown: int) -> None:
    """4 is inside the id range; 10**9 is past the last id, where searchsorted clamps."""
    batch = _peak_batch([3, 7])
    batch["spec_unique_id"] = th.tensor([3, unknown])

    with pytest.raises(ValueError, match=f"not in the JEPA target table.*{unknown}"):
        _lookup(JepaTargetTable(table_fp), batch)


@pytest.mark.parametrize("change", ["rolled_ids", "extra_peak", "dropped_peak", "moved_mz"])
def test_lookup_rejects_peaks_that_are_not_the_tables_raw_peaks(table_fp: str, change: str) -> None:
    table = JepaTargetTable(table_fp)
    ids = [7, 5, 3]
    peaks = [_peaks(spec_id) for spec_id in ids]
    if change == "rolled_ids":
        ids = ids[1:] + ids[:1]
    elif change == "extra_peak":
        peaks[0] = np.append(peaks[0], 500.0)
    elif change == "dropped_peak":
        peaks[0] = peaks[0][:-1]
    else:
        peaks[0] = peaks[0] + np.array([1e-3, 0.0, 0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="do not match the raw peaks"):
        _lookup(table, _peak_batch(ids, peaks))


def test_lookup_accepts_what_float32_rounding_and_a_double_batch_cannot_change(
    table_fp: str,
) -> None:
    table = JepaTargetTable(table_fp)
    batch = _peak_batch([7, 5])
    batch["spec_mzs"] = batch["spec_mzs"].double()  # precision 64 upcasts the batch before the hook

    _, valid = _lookup(table, batch)

    assert valid.tolist() == [True, False]


def test_lookup_requires_cpu_int64_ids(table_fp: str) -> None:
    batch = _peak_batch([7])
    batch["spec_unique_id"] = batch["spec_unique_id"].int()

    with pytest.raises(ValueError, match="spec_ids must be a CPU int64 tensor"):
        _lookup(JepaTargetTable(table_fp), batch)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("schema", "schema"),
        ("missing", "missing arrays"),
        ("unsorted", "strictly ascending"),
        ("duplicate", "strictly ascending"),
        ("emb_float32", "float16"),
        ("emb_width", "float16"),
        ("has_target_int8", "bool"),
        ("nan", "non-finite"),
        ("nonzero_without_target", "nonzero exactly where"),
        ("zero_with_target", "nonzero exactly where"),
    ],
)
def test_table_rejects_a_malformed_file(
    tmp_path: Path, table_fp: str, mutation: str, message: str
) -> None:
    with np.load(table_fp) as good:
        arrays = {key: good[key] for key in good.files}
    if mutation == "schema":
        arrays["schema"] = np.array(2)
    elif mutation == "missing":
        del arrays["mz_sum"]
    elif mutation == "unsorted":
        arrays["spec_id"] = arrays["spec_id"][::-1].copy()
    elif mutation == "duplicate":
        arrays["spec_id"][1] = arrays["spec_id"][0]
    elif mutation == "emb_float32":
        arrays["emb"] = arrays["emb"].astype(np.float32)
    elif mutation == "emb_width":
        arrays["emb"] = arrays["emb"][:, :512].copy()
    elif mutation == "has_target_int8":
        arrays["has_target"] = arrays["has_target"].astype(np.int8)
    elif mutation == "nan":
        arrays["emb"][0, 0] = np.nan
    elif mutation == "nonzero_without_target":
        arrays["emb"][1, 0] = 1.0
    else:
        arrays["emb"][0] = 0
    bad_fp = tmp_path / "bad.npz"
    np.savez(bad_fp, **arrays)

    with pytest.raises(ValueError, match=message):
        JepaTargetTable(str(bad_fp))


def test_table_is_opened_once_per_process(table_fp: str) -> None:
    assert load_jepa_target_table(table_fp) is load_jepa_target_table(table_fp)
    assert load_jepa_target_table.cache_info().misses == 1


def test_raw_peak_signature_matches_the_datasets_float32_cast() -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        mzs = np.round(rng.uniform(50.0, 950.0, size=int(rng.integers(3, 300))), 4)
        count, total = raw_peak_signature(mzs)
        # The dataset builds spec_mzs with th.tensor(list, dtype=th.float) (base_dataset.py).
        delivered = th.tensor(mzs.tolist(), dtype=th.float)
        assert count == delivered.numel()
        assert total == pytest.approx(float(delivered.to(th.float64).sum()), abs=1e-9)


def test_segment_mean_matches_a_loop_and_zeroes_an_empty_segment() -> None:
    th.manual_seed(0)
    values = th.randn(50, 7)
    segments = th.arange(50) % 5

    pooled = _segment_mean(values, segments, 6)  # segment 5 is empty

    for segment in range(5):
        th.testing.assert_close(pooled[segment], values[segments == segment].mean(0))
    assert pooled.shape == (6, 7) and not pooled[5].any()


@pytest.mark.skipif(not th.cuda.is_available(), reason="the sync detector needs CUDA")
def test_pooling_and_the_masked_cosine_add_no_host_device_sync() -> None:
    """th.bincount would synchronize here; the pooling counts with an index_add of ones."""
    values = th.randn(100, 8, device="cuda", requires_grad=True)
    segments = th.arange(100, device="cuda") % 10
    target = th.randn(10, 8, device="cuda").half()
    target[3] = float("nan")
    valid = th.ones(10, dtype=th.bool, device="cuda")
    valid[3] = False
    th.cuda.synchronize()

    th.cuda.set_sync_debug_mode("error")
    try:
        distance, _ = masked_cosine_distance(_segment_mean(values, segments, 10), target, valid)
        distance.sum().backward()
    finally:
        th.cuda.set_sync_debug_mode("default")

    assert th.isfinite(values.grad).all()


def test_masked_cosine_is_exactly_zero_with_zero_gradient_where_masked() -> None:
    pred = th.randn(3, 6, requires_grad=True)
    target = th.randn(3, 6)
    target[1] = float("nan")
    valid = th.tensor([True, False, True])

    distance, cosine = masked_cosine_distance(pred, target, valid)
    distance.sum().backward()

    assert distance[1].item() == 0.0 and cosine[1].item() == 0.0
    expected = 1.0 - F.cosine_similarity(pred[0], target[0], dim=0)
    assert distance[0].item() == pytest.approx(expected.item())
    assert th.isfinite(pred.grad).all()
    assert pred.grad[1].abs().sum().item() == 0.0 and pred.grad[0].abs().sum().item() > 0.0


def test_masked_cosine_casts_a_half_target_and_checks_shapes() -> None:
    pred, valid = th.randn(2, 6), th.tensor([True, True])
    target = th.randn(2, 6).half()

    distance, _ = masked_cosine_distance(pred, target, valid)

    th.testing.assert_close(distance, 1.0 - F.cosine_similarity(pred, target.float(), dim=1))
    with pytest.raises(ValueError, match="must be"):
        masked_cosine_distance(pred, target, th.tensor([True]))


def test_predictor_maps_to_the_target_width_and_draws_no_random_numbers() -> None:
    predictor = build_jepa_predictor(10, 6).train()
    x = th.randn(4, 10)
    rng = th.get_rng_state()

    out = predictor(x)

    assert out.shape == (4, JEPA_EMBEDDING_DIM)
    assert th.equal(th.get_rng_state(), rng)


# 2. Default off, and what enabling adds.


def test_default_builds_nothing_and_registers_no_name() -> None:
    pl = _selection_pl()

    assert pl.hparams.masster_jepa_loss_weight == 0.0
    assert pl.hparams.masster_jepa_source == "encoder"
    assert pl.hparams.masster_jepa_target_fp is None
    assert pl.jepa_predictor is None
    assert not any("jepa" in key for key in pl.state_dict())
    assert JEPA_NAMES.isdisjoint(pl.loss_names) and JEPA_NAMES.isdisjoint(pl.metric_names)


def test_enabling_adds_only_the_predictor_and_leaves_init_and_rng_untouched() -> None:
    off = _seeded(_selection_pl)
    off_rng = th.get_rng_state()
    on = _seeded(_jepa_pl)
    on_rng = th.get_rng_state()

    added = set(on.state_dict()) - set(off.state_dict())
    assert added and all(key.startswith(JEPA_PREFIX) for key in added)
    assert set(off.state_dict()) <= set(on.state_dict())
    assert all(th.equal(tensor, on.state_dict()[key]) for key, tensor in off.state_dict().items())
    assert th.equal(off_rng, on_rng)
    assert set(on.loss_names) >= JEPA_NAMES and on.metric_names >= JEPA_NAMES
    # LayerNorm, Linear(in, hidden), SiLU, Linear(hidden, 1024). The encoder source reads
    # mol_hidden_size + masster_context_dim, the allocator source the transformer width.
    hparams = on.hparams
    assert on.jepa_predictor[1].in_features == hparams.mol_hidden_size + hparams.masster_context_dim
    assert on.jepa_predictor[3].out_features == JEPA_EMBEDDING_DIM
    allocator = _jepa_pl("allocator")
    assert allocator.jepa_predictor[1].in_features == hparams.masster_intensity_transformer_dim


def test_inert_keys_at_weight_zero_change_nothing() -> None:
    plain = _seeded(_selection_pl).train()
    inert = _seeded(
        _selection_pl,
        masster_jepa_source="allocator",
        masster_jepa_target_fp="/nonexistent/table.npz",
        masster_jepa_predictor_hidden_dim=7,
    ).train()
    assert list(inert.state_dict()) == list(plain.state_dict())

    results = []
    for pl in (plain, inert):
        th.manual_seed(5)
        result = pl.loss_fn(**_loss_inputs(pl.forward(**_escape_inputs())))
        result["loss"].sum().backward()
        weight = pl.model.precursor_encoder.input_project.weight
        results.append((result["loss"].detach(), weight.grad))

    th.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    th.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)


def test_a_target_path_without_a_weight_warns_and_stays_off(caplog) -> None:
    with caplog.at_level("WARNING"):
        pl = _selection_pl(masster_jepa_target_fp="/nonexistent/table.npz")

    assert pl.jepa_predictor is None and "the JEPA term is off" in caplog.text
    batch = {"spec_unique_id": th.tensor([7, 5])}
    assert pl.on_before_batch_transfer(batch, 0) is batch


def test_construction_never_opens_the_table() -> None:
    pl = _jepa_pl(fp="/nonexistent/table.npz")

    assert pl.jepa_predictor is not None
    assert load_jepa_target_table.cache_info().currsize == 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"weight": -0.1}, "nonnegative"),
        ({"weight": float("nan")}, "finite"),
        ({"weight": True}, "must be a number"),
        ({"fp": None}, "requires masster_jepa_target_fp"),
        ({"fp": 7}, "path or null"),
        ({"masster_proposal_only_until_epoch": 1}, "masster_proposal_only_until_epoch"),
        ({"eval_test_split": True}, "eval_test_split"),
    ],
)
def test_setup_rejects_what_the_term_cannot_work_with(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _jepa_pl(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"masster_jepa_source": "decoder"},
        {"masster_jepa_predictor_hidden_dim": 0},
        {"masster_jepa_predictor_hidden_dim": True},
        {"masster_jepa_target_fp": 7},
        {"masster_jepa_loss_weight": "x"},
    ],
)
def test_the_keys_are_checked_even_while_the_term_is_off(overrides: dict) -> None:
    with pytest.raises(ValueError):
        _selection_pl(**overrides)


def test_setup_requires_unmerged_spectra_with_ids() -> None:
    spec_params = _i1_like()["spec_params"]
    for changes in ({"unique_id": False}, {"merge": True, "merge_keep_ces": True}):
        with pytest.raises(ValueError, match="merge: false and unique_id: true"):
            _jepa_pl(spec_params={**spec_params, **changes})


def test_allocator_source_requires_unique_fragment_mode() -> None:
    with pytest.raises(ValueError, match="requires masster_unique_fragment_mode"):
        _build_pl(
            **_i1_like(),
            masster_jepa_loss_weight=1.0,
            masster_jepa_source="allocator",
            masster_jepa_target_fp="unused.npz",
        )


def test_a_checkpoint_loads_only_into_a_module_with_the_same_predictor() -> None:
    off, on = _seeded(_selection_pl), _seeded(_jepa_pl)

    with pytest.raises(RuntimeError, match="jepa_predictor"):
        off.load_state_dict(on.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="jepa_predictor"):
        on.load_state_dict(off.state_dict(), strict=True)
    _seeded(_jepa_pl).load_state_dict(on.state_dict(), strict=True)


@pytest.mark.parametrize("exclude_norm_bias", [False, True])
def test_every_predictor_parameter_is_in_the_optimizer(exclude_norm_bias: bool) -> None:
    pl = _jepa_pl(weight_decay_exclude_norm_bias=exclude_norm_bias)

    optimizer = pl.configure_optimizers()["optimizer"]

    in_optimizer = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert pl.jepa_predictor is not None
    assert all(id(p) in in_optimizer for p in pl.jepa_predictor.parameters())


# 3. The pooled representations.


def test_head_pooled_tokens_are_the_per_spectrum_mean_and_change_no_output() -> None:
    args, kwargs = _head_inputs()
    head = _head(predict_oos=True).eval()
    tokens = {}
    head.component_residual_head.register_forward_pre_hook(
        lambda module, inputs: tokens.setdefault("value", inputs[0])
    )
    plain = head(*args, **kwargs)
    outputs, pooled = head(*args, **kwargs, return_pooled_tokens=True)

    assert len(outputs) == len(plain) == 4
    assert all(th.equal(a, b) for a, b in zip(outputs, plain, strict=True))
    expected = th.stack([tokens["value"][args[5] == index].mean(0) for index in (0, 1)])
    assert pooled.shape == (2, 8)
    th.testing.assert_close(pooled, expected)
    # Without the escape head the pooled call is (outputs, pooled) too, and the plain call is
    # the bare three-tuple it always was.
    bare = _head().eval()
    bare_outputs, bare_pooled = bare(*args, **kwargs, return_pooled_tokens=True)
    assert len(bare(*args, **kwargs)) == len(bare_outputs) == 3
    th.testing.assert_close(bare_pooled, pooled)


def _selection_only_model(smiles: tuple[str, ...] = ("CCOC", "CCO"), **kwargs):
    batch = _build_batch(smiles)
    return batch, _build_model(
        batch,
        max_masks=4,
        unique_fragment_mode=True,
        max_enumerated_masks=8,
        use_base_probability_feature=False,
        proposal_selection_only=True,
        **kwargs,
    )


@pytest.mark.parametrize("source", ["encoder", "allocator"])
def test_model_returns_the_pooled_vector_and_changes_no_other_output(source: str) -> None:
    batch, model = _selection_only_model()
    model.eval()
    captured = {}
    model.precursor_encoder.register_forward_hook(
        lambda module, inputs, out: captured.__setitem__("atoms", out)
    )
    model.context_encoder.register_forward_hook(
        lambda module, inputs, out: captured.__setitem__("context", out)
    )
    head = model.intensity_head
    head.component_residual_head.register_forward_pre_hook(
        lambda module, inputs: captured.__setitem__("tokens", inputs[0])
    )
    head.register_forward_pre_hook(lambda module, inputs: captured.__setitem__("rows", inputs[5]))
    with th.no_grad():
        plain = _forward(model, batch)
        pooled = _forward(model, batch, jepa_source=source)

    assert set(pooled) - set(plain) == {"jepa_pooled"} and set(plain) <= set(pooled)
    for key, value in plain.items():
        # The candidate-generation timings differ between any two calls.
        if isinstance(value, th.Tensor) and not key.endswith("_ms"):
            th.testing.assert_close(pooled[key], value, rtol=0, atol=0, equal_nan=True)
    if source == "encoder":
        atoms = th.stack([captured["atoms"][batch.batch == i].mean(0) for i in (0, 1)])
        assert pooled["jepa_pooled"].shape == (2, 16 + 8)
        th.testing.assert_close(pooled["jepa_pooled"][:, :16], atoms)
        th.testing.assert_close(pooled["jepa_pooled"][:, 16:], captured["context"])
    else:
        tokens = th.stack([captured["tokens"][captured["rows"] == i].mean(0) for i in (0, 1)])
        assert pooled["jepa_pooled"].shape == (2, 16)
        th.testing.assert_close(pooled["jepa_pooled"], tokens)


def test_model_rejects_an_unknown_source_and_an_allocator_without_unique_fragments() -> None:
    batch = _build_batch(("CCO", "CO"))
    model = _build_model(batch)

    with pytest.raises(ValueError, match="None, 'encoder' or 'allocator'"):
        _forward(model, batch, jepa_source="decoder")
    with pytest.raises(ValueError, match="requires unique-fragment mode"):
        _forward(model, batch, jepa_source="allocator")


@pytest.mark.parametrize("source", ["encoder", "allocator"])
def test_the_term_alone_reaches_only_the_modules_it_reads(source: str) -> None:
    batch = _build_batch(("CCCO",))
    model = _build_model(
        batch,
        max_masks=5,
        unique_fragment_mode=True,
        max_enumerated_masks=16,
        use_base_probability_feature=False,
        proposal_selection_only=True,
    )
    output = model(
        mol_pyg=batch, spec_prec_type=th.tensor([0]), spec_nce=th.tensor([25.0]), jepa_source=source
    )

    output["jepa_pooled"].pow(2).sum().backward()

    assert _parameter_gradient_sum(model.precursor_encoder) > 0.0
    assert _parameter_gradient_sum(model.bond_break_head) == 0.0
    assert _parameter_gradient_sum(model.charge_head) == 0.0
    if source == "encoder":
        assert _parameter_gradient_sum(model.context_encoder) > 0.0
        assert _parameter_gradient_sum(model.intensity_head) == 0.0
    else:
        assert _parameter_gradient_sum(model.intensity_head) > 0.0


def test_shared_proposal_layout_still_pools_one_vector_per_spectrum() -> None:
    """Three spectra, two of one molecule: the atom half is shared, the context half is not."""
    batch = _build_batch(("CCCO", "CCCO", "CCCC"))
    model = _build_model(
        batch,
        max_masks=4,
        unique_fragment_mode=True,
        max_enumerated_masks=12,
        use_base_probability_feature=False,
        proposal_selection_only=True,
        separate_proposal_encoder=True,
        share_proposal_across_ev=True,
    ).eval()

    with th.no_grad():
        output = model(
            mol_pyg=batch,
            spec_prec_type=th.zeros(3, dtype=th.long),
            spec_nce=th.tensor([10.0, 40.0, 10.0]),
            proposal_group_ids=th.tensor([3, 3, 5]),
            jepa_source="encoder",
        )

    pooled = output["jepa_pooled"]
    assert pooled.shape == (3, 16 + 8)
    th.testing.assert_close(pooled[0, :16], pooled[1, :16])
    assert not th.allclose(pooled[0, 16:], pooled[1, 16:])
    assert not th.allclose(pooled[2, :16], pooled[0, :16])


# 4. The Lightning wrapper: hook, forward, loss.


def test_hook_attaches_targets_only_when_on_and_leaves_its_input_alone(table_fp: str) -> None:
    off = _selection_pl(masster_jepa_target_fp="/nonexistent/table.npz")
    plain = {"spec_unique_id": th.tensor([7, 5])}
    assert off.on_before_batch_transfer(plain, 0) is plain

    on = _jepa_pl(fp=table_fp)
    batch = _peak_batch([7, 5])
    hooked = on.on_before_batch_transfer(batch, 0)

    assert set(batch) == {"spec_unique_id", "spec_mzs", "spec_batch_idxs"}
    assert hooked["jepa_valid"].tolist() == [True, False]
    assert hooked["jepa_target"].dtype == th.float16 and hooked["jepa_target"].shape == (2, 1024)
    assert not hooked["jepa_target"][1].any()
    assert all(hooked[key] is batch[key] for key in batch)
    unknown = _peak_batch([4, 3])
    with pytest.raises(ValueError, match="not in the JEPA target table.*4"):
        on.on_before_batch_transfer(unknown, 0)
    stale = _peak_batch([7, 5], [_peaks(7) + 1.0, _peaks(5)])
    with pytest.raises(ValueError, match="do not match the raw peaks"):
        on.on_before_batch_transfer(stale, 0)


def test_forward_asks_for_the_pool_only_when_the_batch_carries_a_target(table_fp: str) -> None:
    on = _jepa_pl(fp=table_fp).eval()
    inputs = _escape_inputs()
    batch = _hooked(on, [7, 5])

    with th.no_grad():
        without = on.forward(**inputs)
        predicted = on.predict_step(**inputs)
        with_target = on.forward(**batch)
        off_output = _selection_pl().eval().forward(**batch)

    assert not any("jepa" in key for key in without) and not any("jepa" in k for k in predicted)
    hparams = on.hparams
    assert with_target["jepa_pooled"].shape == (
        2,
        hparams.mol_hidden_size + hparams.masster_context_dim,
    )
    assert th.equal(with_target["jepa_target"], batch["jepa_target"])
    assert th.equal(with_target["jepa_valid"], batch["jepa_valid"])
    assert not any("jepa" in key for key in off_output)


def test_loss_adds_the_term_in_training_only_and_logs_it_in_both(table_fp: str) -> None:
    on = _seeded(_jepa_pl, fp=table_fp)
    off = _seeded(_selection_pl)
    batch = _hooked(on, [7, 5])
    on.train()
    loss_inputs = _loss_inputs(on.forward(**batch))
    plain_inputs = {k: v for k, v in loss_inputs.items() if not k.startswith("jepa")}

    trained = on.loss_fn(**loss_inputs)
    on.eval()
    evaluated = on.loss_fn(**loss_inputs)
    reference = off.loss_fn(**plain_inputs)

    assert set(trained) >= JEPA_NAMES and set(evaluated) >= JEPA_NAMES
    assert trained["jepa_loss"].shape == (2,) and trained["jepa_loss"][1].item() == 0.0
    assert trained["jepa_target_fraction"].tolist() == [1.0, 0.0]
    cosine = trained["jepa_cos"][0].item()
    assert cosine == pytest.approx(1.0 - trained["jepa_loss"][0].item(), abs=1e-6)
    assert th.equal(trained["jepa_loss"], evaluated["jepa_loss"])
    assert not trained["jepa_loss"].requires_grad and not trained["jepa_cos"].requires_grad
    # The term is the only difference in training, and it is absent from the validation loss.
    th.testing.assert_close(trained["loss"] - reference["loss"], 0.5 * trained["jepa_loss"])
    th.testing.assert_close(evaluated["loss"], reference["loss"], rtol=0, atol=0)
    assert th.equal(trained["primary_loss"], reference["primary_loss"])
    trained["loss"].sum().backward()
    grads = [p.grad for p in on.jepa_predictor.parameters()]
    assert all(g is not None and th.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0.0


def test_loss_without_a_target_is_zeros_in_eval_and_an_error_in_training(table_fp: str) -> None:
    on = _seeded(_jepa_pl, fp=table_fp).eval()
    off = _seeded(_selection_pl).eval()
    inputs = _loss_inputs(on.forward(**_escape_inputs()))

    result = on.loss_fn(**inputs)

    assert set(on.loss_names) <= set(result)
    for name in JEPA_NAMES:
        assert result[name].shape == (2,) and not result[name].any()
    th.testing.assert_close(result["loss"], off.loss_fn(**inputs)["loss"], rtol=0, atol=0)
    on.train()
    with pytest.raises(ValueError, match="no jepa_target.*on_before_batch_transfer"):
        on.loss_fn(**inputs)


def test_the_structure_is_fixed_at_setup_and_only_the_weight_is_read_later(
    table_fp: str,
) -> None:
    """A weight set to 0 after setup keeps hook, pooling and logging and adds exactly zero."""
    on = _seeded(_jepa_pl, fp=table_fp).train()
    off = _seeded(_selection_pl).train()
    on.hparams.masster_jepa_loss_weight = 0.0
    batch = _hooked(on, [7, 5])

    assert "jepa_target" in batch
    th.manual_seed(5)
    result = on.loss_fn(**_loss_inputs(on.forward(**batch)))
    th.manual_seed(5)
    reference = off.loss_fn(**_loss_inputs(off.forward(**_escape_inputs())))

    assert result["jepa_loss"][0] > 0 and result["jepa_target_fraction"].tolist() == [1.0, 0.0]
    th.testing.assert_close(result["loss"], reference["loss"], rtol=0, atol=0)


def test_inference_outputs_never_hold_a_jepa_key(table_fp: str) -> None:
    """Standalone inference never runs the hook, so its batches carry no target."""
    on = _jepa_pl(fp=table_fp).eval()

    with th.no_grad():
        result = on.inference_step(dict(_escape_inputs()))

    assert not any("jepa" in key for key in result)


SMILES = ["c1ccccc1S(=O)c1ccccc1", "CCOC(=O)C", "c1ccccc1O", "CC(C)NCC(O)c1ccc(O)cc1"]
NCE = [20.0, 35.0, 25.0, 45.0]
IDS = [11, 4207, 90001, 555]


def _molecule_batch(pl: MassterPL, config: dict, order: list[int]) -> dict:
    """A batch of distinct molecules at distinct energies, in the given order, with ids and peaks."""
    params = config["mol_params"]
    graphs = [
        get_mol_graph(
            Chem.MolFromSmiles(SMILES[i]),
            params["pyg_node_feats"],
            params["pyg_edge_feats"],
            params["pyg_pe_embed_k"],
            True,
        )
        for i in order
    ]
    return {
        "mol_pyg": Batch.from_data_list(graphs),
        "spec_prec_type": th.zeros(len(order), dtype=th.long),
        "spec_inst_type": th.zeros(len(order), dtype=th.long),
        "spec_nce": th.tensor([NCE[i] for i in order]),
        **_peak_batch([IDS[i] for i in order]),
    }


@pytest.mark.parametrize("source", ["encoder", "allocator"])
def test_hook_forward_and_loss_pair_every_row_with_its_own_target(
    tmp_path: Path, source: str
) -> None:
    """Targets come from singleton forwards; shuffled batches must score cosine 1 on every row.

    A target that landed on the wrong row, a pooled row in the wrong order, or a mean over the
    wrong spectra would each break this. The negative control rolls the targets by one row.
    """
    th.manual_seed(0)
    path = str(tmp_path / "self_consistent.npz")
    pl, config = _build_selection_only_pl(
        **_i1_like(),
        masster_jepa_loss_weight=1.0,
        masster_jepa_source=source,
        masster_jepa_target_fp=path,
        masster_jepa_predictor_hidden_dim=32,
    )
    pl.eval()
    with th.no_grad():
        pl.jepa_predictor[1].weight.mul_(4.0)  # more input-specific hidden units

    def pooled_of(order: list[int]) -> th.Tensor:
        """The pooled representation of a batch, from a forward that skips the hook."""
        batch = _molecule_batch(pl, config, order)
        batch["jepa_target"] = th.zeros(len(order), JEPA_EMBEDDING_DIM, dtype=th.float16)
        batch["jepa_valid"] = th.ones(len(order), dtype=th.bool)
        with th.no_grad():
            return pl.forward(**batch)["jepa_pooled"]

    with th.no_grad():
        targets = th.stack([pl.jepa_predictor(pooled_of([index]))[0] for index in range(4)])
    signatures = [raw_peak_signature(_peaks(spec_id)) for spec_id in IDS]
    write_jepa_table(
        path,
        spec_id=np.array(IDS),
        has_target=np.ones(4, dtype=bool),
        emb=targets.half().numpy(),
        n_peaks=np.array([s[0] for s in signatures]),
        mz_sum=np.array([s[1] for s in signatures]),
        source="self-consistency",
    )
    # A random predictor maps different molecules to nearby directions; they must still differ.
    pair_distance = (
        1.0 - (F.cosine_similarity(targets[:, None], targets[None], dim=2) - 2 * th.eye(4)).max()
    )
    assert pair_distance > 1e-4

    for order in ([2, 0, 3], [3, 1, 0], [1, 2, 3]):
        batch = pl.on_before_batch_transfer(_molecule_batch(pl, config, order), 0)
        with th.no_grad():
            output = pl.forward(**batch)
            aligned = pl.loss_fn(**_loss_inputs(output))
            output["jepa_target"] = batch["jepa_target"].roll(1, 0)
            rolled = pl.loss_fn(**_loss_inputs(output))
        assert aligned["jepa_target_fraction"].tolist() == [1.0, 1.0, 1.0]
        assert (1.0 - aligned["jepa_cos"]).max() < 1e-5, (source, order)
        assert (1.0 - rolled["jepa_cos"]).max() > 1e-4, (source, order)


def test_lightning_loops_call_the_hook_with_cpu_ids_and_move_aligned_targets(
    tmp_path: Path,
) -> None:
    """Sanity check, training and validation each see the table rows of their own ids."""
    seen: list[tuple] = []

    class Probe(MassterPL):
        def _record(self, tag: str, batch: dict) -> None:
            seen.append(
                (
                    tag,
                    self.training,
                    batch["spec_unique_id"].tolist(),
                    batch["jepa_target"].dtype,
                    batch["jepa_target"].device.type,
                    batch["jepa_valid"].tolist(),
                    batch["jepa_target"].float().sum(dim=1).ne(0).tolist(),
                )
            )

        def training_step(self, batch, batch_idx):
            self._record("train", batch)
            return self.probe_param.sum() * 0.0 + 1.0

        def validation_step(self, batch, batch_idx):
            self._record("val", batch)

    th.manual_seed(0)
    pl0, config = _build_selection_only_pl(
        **_i1_like(),
        masster_jepa_loss_weight=0.5,
        masster_jepa_target_fp=_write_table(tmp_path / "table.npz"),
    )
    probe = Probe(**config)
    probe.probe_param = th.nn.Parameter(th.zeros(1))
    # The static graph is added by transfer_batch_to_device, after the hook, in a real batch.
    collated = {k: v for k, v in _escape_inputs().items() if k != "static_graph"}
    batch = {**collated, **_peak_batch([20, 9]), "mol_smiles": ["c1ccccc1S(=O)c1ccccc1"] * 2}
    loader = DataLoader([batch], batch_size=None, collate_fn=lambda item: item)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=1,
    )

    trainer.fit(probe, loader, loader)

    assert [entry[:2] for entry in seen] == [("val", False), ("train", True), ("val", False)]
    for _, _, ids, dtype, device, valid, nonzero in seen:
        assert ids == [20, 9] and dtype == th.float16 and device == "cpu"
        assert valid == [True, False] and nonzero == [True, False]
