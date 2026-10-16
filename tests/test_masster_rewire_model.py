"""Model wiring of the MASSter rewire-rule candidates (build step 3, arm a0)."""

import copy
import math

import pytest
import torch as th
from rdkit import Chem

import fragnnet.models.masster.masster_model as masster_model_module
from fragnnet.models.masster.candidates.builder import MassterStaticGraph
from fragnnet.models.masster.candidates.rewire_rules import (
    MASSTER_NUM_REWIRE_RULES,
    REWIRE_RULE_SPEC_HASHES,
    REWIRE_RULE_VOCABULARY,
    build_rewire_matches,
)
from fragnnet.models.masster.candidates.smarts_templates import build_masster_templates
from fragnnet.models.masster.masster_pl import _channel_support_metrics, _proposal_coverage_loss
from tests.test_masster_model import _build_batch, _build_model

_SULFONE = "O=S(=O)(c1ccccc1)c1ccccc1"
_METHYL_SULFONE = "CS(=O)(=O)c1ccccc1"
_DIBENZOTHIOPHENE_DIOXIDE = "O=S1(=O)c2ccccc2-c2ccccc12"
_SULFONAMIDE = "CC(C)(C)c1ccc(cc1)S(=O)(=O)Nc1ccccc1"
_AZOBENZENE = "c1ccccc1N=Nc1ccccc1"

_REWIRE_KWARGS = dict(
    unique_fragment_mode=True,
    proposal_selection_only=True,
    use_base_probability_feature=False,
    h_transfer_bound="valence",
    max_masks=8,
    max_enumerated_masks=32,
    use_rewire_rules=True,
    rewire_max_extra_fraction=1.0,
    intensity_transformer_attention="none",
)
_OFF_KWARGS = {
    key: value
    for key, value in _REWIRE_KWARGS.items()
    if not key.startswith("rewire_") and key != "use_rewire_rules"
}


def _static_graph(batch, smiles, *, with_matches=True, smarts=False) -> MassterStaticGraph:
    return MassterStaticGraph(
        chemical_bond_index=batch.chemical_bond_index,
        atom_batch_indices=batch.batch,
        atom_ptr=batch.ptr,
        smarts_templates=(
            tuple(build_masster_templates(Chem.MolFromSmiles(s)) for s in smiles)
            if smarts
            else None
        ),
        rewire_matches=(
            tuple(
                build_rewire_matches(Chem.MolFromSmiles(s), max_matches_per_rule=8) for s in smiles
            )
            if with_matches
            else None
        ),
    )


def _run(model, batch, smiles, *, prec=None, with_matches=True, smarts=False, **kwargs):
    prec = th.zeros(len(smiles), dtype=th.long) if prec is None else prec
    return model(
        mol_pyg=batch,
        spec_prec_type=prec,
        spec_nce=th.full((len(smiles),), 30.0),
        static_graph=_static_graph(batch, smiles, with_matches=with_matches, smarts=smarts),
        **kwargs,
    )


def _pair(smiles, **extra):
    batch = _build_batch(tuple(smiles))
    off = _build_model(batch, **_OFF_KWARGS, **{k: v for k, v in extra.items() if k != "on"})
    on_kwargs = {**_REWIRE_KWARGS, **extra.get("on", {})}
    on = _build_model(batch, **{k: v for k, v in on_kwargs.items()}, **{
        k: v for k, v in extra.items() if k != "on"
    })
    return batch, off, on


def _same(a, b):
    return a.shape == b.shape and th.equal(a, b)


# --- OFF identity ----------------------------------------------------------------------------


def test_off_model_has_no_rewire_state() -> None:
    batch = _build_batch((_SULFONE,))
    model = _build_model(batch, **_OFF_KWARGS)
    explicit = _build_model(batch, **_OFF_KWARGS, use_rewire_rules=False)
    assert not [key for key in model.state_dict() if "rewire" in key]
    assert model.intensity_head.rewire_pathway_residual is None
    assert not model.use_rewire_rules
    assert model.state_dict().keys() == explicit.state_dict().keys()
    for key, value in model.state_dict().items():
        assert th.equal(value, explicit.state_dict()[key])


def test_off_forward_ignores_rewire_matches() -> None:
    batch = _build_batch((_SULFONE, "CCO"))
    smiles = (_SULFONE, "CCO")
    model = _build_model(batch, **_OFF_KWARGS).eval()
    with th.no_grad():
        with_matches = _run(model, batch, smiles)
        without = _run(model, batch, smiles, with_matches=False)
    assert with_matches.keys() == without.keys()
    assert not [key for key in with_matches if "rewire" in key]
    for key, value in with_matches.items():
        if isinstance(value, th.Tensor) and "timing" not in key and not key.endswith("_ms"):
            assert _same(value, without[key]), key


def test_on_model_starts_as_the_control_parameters() -> None:
    batch, off, on = _pair((_SULFONE,))
    on_state = on.state_dict()
    for key, value in off.state_dict().items():
        assert th.equal(value, on_state[key]), key
    assert set(on_state) - set(off.state_dict()) == {
        "rewire_rule_name_hashes",
        "intensity_head.rewire_pathway_residual.weight",
    }
    assert not on_state["intensity_head.rewire_pathway_residual.weight"].any()
    assert on_state["intensity_head.rewire_pathway_residual.weight"].shape[1] == (
        MASSTER_NUM_REWIRE_RULES
    )


def test_on_without_matching_molecules_equals_control() -> None:
    smiles = ("CCO", "CO")
    batch, off, on = _pair(smiles)
    off.eval()
    on.eval()
    with th.no_grad():
        control = _run(off, batch, smiles)
        out = _run(on, batch, smiles)
    assert not [key for key in control if key not in out]
    assert out["rewire_is_extra"].numel() == out["ion_h_fragment_batch_idxs"].numel()
    assert not out["rewire_is_extra"].any()
    for key, value in control.items():
        if isinstance(value, th.Tensor) and "timing" not in key and not key.endswith("_ms"):
            assert _same(value, out[key]), key
    assert out["rewire_match_count"].tolist() == [0.0, 0.0]


def test_on_without_matches_equals_control_at_default_attention() -> None:
    smiles = ("CCO", "CO")
    batch = _build_batch(smiles)
    kwargs = {k: v for k, v in _REWIRE_KWARGS.items() if k != "intensity_transformer_attention"}
    off = _build_model(
        batch, **{k: v for k, v in kwargs.items() if not k.startswith("rewire_") and k != "use_rewire_rules"}
    ).eval()
    on = _build_model(batch, **kwargs).eval()
    with th.no_grad():
        control = _run(off, batch, smiles)
        out = _run(on, batch, smiles)
    for key in ("pred_mzs", "pred_logprobs", "pred_batch_idxs", "ion_h_dense_log_weights"):
        assert _same(control[key], out[key]), key


@pytest.mark.parametrize("light_side", [False, True])
def test_light_side_token_and_heavy_side_judge_the_product_mass(light_side) -> None:
    smiles = (_SULFONE,)
    batch = _build_batch(smiles)
    kwargs = {**_REWIRE_KWARGS, "candidate_features": ("c_cut_count", "c_size", "c_heavy_side")}
    if light_side:
        kwargs["light_side_token"] = True
    model = _build_model(batch, **kwargs)
    out = _run(model, batch, smiles)
    assert out["rewire_is_extra"].any() and th.isfinite(out["pred_logprobs"]).all()


# --- guards ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"unique_fragment_mode": False, "proposal_selection_only": False}, "unique_fragment_mode"),
        (
            {"fragment_selection": "marginal", "selected_fragment_target": 4},
            "mask_order",
        ),
        ({"proposal_selection_only": False}, "proposal_selection_only"),
        ({"h_transfer_bound": "none"}, "h_transfer_bound"),
        ({"proposal_method": "connected_fragment_beam"}, "cut_masks"),
        ({"fragment_connectivity_layers": 1}, "fragment_connectivity_layers"),
        (
            {
                "intensity_candidate_graph": True,
                "intensity_transformer_attention": "full",
            },
            "intensity_candidate_graph",
        ),
        (
            {"separate_ion_mode_proposal_head": True},
            "separate_ion_mode_proposal_head",
        ),
        ({"rewire_max_extra_fraction": 0.0}, "rewire_max_extra_fraction"),
        ({"rewire_max_extra_fraction": float("nan")}, "rewire_max_extra_fraction"),
        ({"rewire_max_extra": 0}, "rewire_max_extra"),
        ({"rewire_max_extra": True}, "rewire_max_extra"),
        ({"rewire_continuation_cuts": -1}, "rewire_continuation_cuts"),
        ({"rewire_pair_chunk": 0}, "rewire_pair_chunk"),
        ({"use_rewire_rules": 1}, "use_rewire_rules must be boolean"),
    ],
)
def test_constructor_guards(override: dict, message: str) -> None:
    batch = _build_batch((_SULFONE,))
    with pytest.raises(ValueError, match=message):
        _build_model(batch, **{**_REWIRE_KWARGS, **override})


def test_guard_return_base_spectrum_and_base_feature() -> None:
    batch = _build_batch((_SULFONE,))
    with pytest.raises(ValueError, match="return_base_spectrum"):
        _build_model(batch, **{**_REWIRE_KWARGS, "return_base_spectrum": True})
    with pytest.raises(ValueError):
        _build_model(batch, **{**_REWIRE_KWARGS, "use_base_probability_feature": True})


def test_edit_descriptor_requires_the_flag() -> None:
    batch = _build_batch((_SULFONE,))
    with pytest.raises(ValueError, match="c_rewire_edit"):
        _build_model(batch, **_OFF_KWARGS, candidate_features=("c_cut_count", "c_rewire_edit"))


def test_forward_requires_rewire_matches() -> None:
    batch, _, on = _pair((_SULFONE,))
    with pytest.raises(ValueError, match="rewire_matches"):
        _run(on, batch, (_SULFONE,), with_matches=False)


# --- placement and proposal parity ---------------------------------------------------------


def _proposal_grad(out):
    loss = (out["proposal_fragment_logprobs"] * th.linspace(0.5, 1.5, out["proposal_fragment_logprobs"].numel())).sum()
    return th.autograd.grad(loss, out["bond_break_logits"], allow_unused=True)[0]


@pytest.mark.parametrize("smiles", [(_SULFONE,), (_SULFONE, "CCO"), ("CCO", _METHYL_SULFONE)])
def test_extras_follow_ordinary_rows_and_proposal_outputs_equal_control(smiles) -> None:
    batch, off, on = _pair(smiles)
    control = _run(off, batch, smiles)
    out = _run(on, batch, smiles)
    is_extra = out["rewire_is_extra"]
    assert is_extra.any()
    first = int(th.nonzero(is_extra)[0])
    assert is_extra[first:].all() and not is_extra[:first].any()
    assert out["rewire_match_count"].sum() > 0
    for key in (
        "proposal_candidate_mzs",
        "proposal_candidate_fragment_idxs",
        "proposal_fragment_logprobs",
        "proposal_fragment_batch_idxs",
        "bond_break_logits",
        "fragment_count_before_cutoff",
        "retained_fragment_probability",
    ):
        assert _same(control[key], out[key]), key
    assert _same(_proposal_grad(control), _proposal_grad(out))
    # Extras are valid by construction and carry a zero proposal probability.
    assert out["ion_h_chemical_valid_channels"][is_extra].flatten(1).any(1).all()
    assert (out["ion_h_fragment_proposal_logprobs"][is_extra].exp() == 0).all()
    assert out["pred_batch_idxs"].numel() > control["pred_batch_idxs"].numel()
    assert th.isfinite(out["pred_logprobs"]).all()
    # Rewire probabilities are normalized per spectrum over ordinary rows and extras.
    totals = th.zeros(len(smiles)).index_add(
        0, out["pred_batch_idxs"], out["pred_logprobs"].exp()
    )
    assert th.allclose(totals, th.ones_like(totals), atol=1e-5)


def test_extra_count_diagnostics_and_node_growth() -> None:
    batch, off, on = _pair((_SULFONE,))
    control = _run(off, batch, (_SULFONE,))
    out = _run(on, batch, (_SULFONE,))
    extras = int(out["rewire_is_extra"].sum())
    assert extras == int(out["rewire_extra_count"].sum())
    assert out["ion_h_fragment_batch_idxs"].numel() == control["ion_h_fragment_batch_idxs"].numel() + extras
    assert float(out["rewire_precap_extra_count"].sum()) == (
        float(out["rewire_dropped_no_valid_count"].sum())
        + float(out["rewire_dropped_cap_count"].sum())
        + extras
    )


def test_filler_log_probability_of_extras_is_unread(monkeypatch) -> None:
    smiles = (_SULFONE,)
    batch, _, on = _pair(smiles)
    on.eval()
    with th.no_grad():
        reference = _run(on, batch, smiles)
        original = masster_model_module.MassterModel._append_rewire_rows

        def patched(self, **kwargs):
            result = original(self, **kwargs)
            filler = result["fragment_log_probabilities"].clone()
            filler[result["rewire_is_extra"]] = 7.5
            result["fragment_log_probabilities"] = filler
            return result

        monkeypatch.setattr(masster_model_module.MassterModel, "_append_rewire_rows", patched)
        changed = _run(on, batch, smiles)
    for key in ("pred_mzs", "pred_logprobs", "ion_h_dense_log_weights"):
        assert _same(reference[key], changed[key]), key


def test_mixed_batch_equals_per_spectrum_runs() -> None:
    smiles = (_SULFONE, "CCO")
    batch, _, on = _pair(smiles)
    on.eval()
    with th.no_grad():
        mixed = _run(on, batch, smiles)
        for index, molecule in enumerate(smiles):
            single_batch = _build_batch((molecule,))
            single = _run(on, single_batch, (molecule,))
            mask = mixed["pred_batch_idxs"] == index
            assert th.allclose(mixed["pred_mzs"][mask], single["pred_mzs"], atol=0, rtol=0)
            assert th.allclose(mixed["pred_logprobs"][mask], single["pred_logprobs"], atol=1e-6)
            row_mask = mixed["ion_h_fragment_batch_idxs"] == index
            assert _same(mixed["rewire_is_extra"][row_mask], single["rewire_is_extra"])


def test_multi_adduct_azo_rule_is_adduct_specific() -> None:
    smiles = (_AZOBENZENE, _AZOBENZENE)
    batch, _, on = _pair(smiles)
    azo = REWIRE_RULE_VOCABULARY.index("azo_N2_extrusion")
    prec = th.tensor([0, 1])  # [M+H]+, [M-H]-
    out = _run(on, batch, smiles, prec=prec, return_annotation=True)
    rows = out["fragment_rewire_fragment_idxs"]
    rule_rows = rows[out["fragment_rewire_rule_idxs"] == azo]
    spectra = out["ion_h_fragment_batch_idxs"][rule_rows]
    assert spectra.numel() > 0 and (spectra == 1).all()


# --- gradients, merge, features -------------------------------------------------------------


def test_intensity_gradient_reaches_only_matched_rule_columns() -> None:
    smiles = (_METHYL_SULFONE,)
    batch, _, on = _pair(smiles)
    out = _run(on, batch, smiles, return_annotation=True)
    weights = th.linspace(0.3, 1.7, out["pred_logprobs"].numel())
    (out["pred_logprobs"] * weights).sum().backward()
    grad = on.intensity_head.rewire_pathway_residual.weight.grad
    assert grad is not None and grad.shape[1] == MASSTER_NUM_REWIRE_RULES
    touched = set(th.nonzero(grad.abs().sum(0)).flatten().tolist())
    matched = set(out["fragment_rewire_rule_idxs"].tolist())
    assert touched and touched <= matched
    assert REWIRE_RULE_VOCABULARY.index("sulfonyl_so2_extrusion") in touched
    # The proposal parameters get no gradient from the intensity loss (arm a0).
    assert all(
        parameter.grad is None or not parameter.grad.any()
        for name, parameter in on.named_parameters()
        if name.startswith("bond_break_head.")
    )


def test_merged_row_gains_indicator_and_channels_without_new_row() -> None:
    smiles = (_DIBENZOTHIOPHENE_DIOXIDE,)
    batch, off, on = _pair(smiles)
    control = _run(off, batch, smiles)
    out = _run(on, batch, smiles, return_annotation=True)
    s_to_o = REWIRE_RULE_VOCABULARY.index("s_to_o_aromatic_C")
    merged = out["fragment_rewire_fragment_idxs"][out["fragment_rewire_rule_idxs"] == s_to_o]
    assert merged.numel() > 0
    # The precursor is an ordinary row (cut count 0) and is not an extra.
    precursor_rows = th.nonzero(out["rewire_is_extra"] == False).flatten()  # noqa: E712
    assert set(merged.tolist()) & set(precursor_rows.tolist())
    assert (~out["rewire_is_extra"][merged[~out["rewire_is_extra"][merged]]]).all()
    # Channels only grow on ordinary rows, and the proposal view is the control's.
    n_ord = control["ion_h_fragment_batch_idxs"].numel()
    assert (
        out["ion_h_chemical_valid_channels"][:n_ord] >= control["ion_h_chemical_valid_channels"]
    ).all()
    for key in ("proposal_candidate_mzs", "proposal_fragment_logprobs", "proposal_candidate_fragment_idxs"):
        assert _same(control[key], out[key]), key
    assert _same(_proposal_grad(control), _proposal_grad(out))


def test_edit_descriptor_hand_row(monkeypatch) -> None:
    smiles = (_SULFONE,)
    batch = _build_batch(smiles)
    model = _build_model(
        batch, **_REWIRE_KWARGS, candidate_features=("c_cut_count", "c_size", "c_rewire_edit")
    )
    captured = {}
    original = masster_model_module.build_candidate_features

    def spy(specs, inputs):
        captured["edit"] = inputs.rewire_edit_features
        return original(specs, inputs)

    monkeypatch.setattr(masster_model_module, "build_candidate_features", spy)
    out = _run(model, batch, smiles)
    edit = captured["edit"]
    is_extra = out["rewire_is_extra"]
    n_ord = int((~is_extra).sum())
    assert edit.shape[0] == is_extra.numel() and not edit[:n_ord].any()
    assert [1.0, 0.0, 3.0, 0.0] in edit[n_ord:].tolist()
    assert edit.shape[1] == 4


def test_edit_descriptor_is_inert_when_unlisted() -> None:
    batch, off, on = _pair((_SULFONE,))
    assert on.candidate_feature_dim == off.candidate_feature_dim


# --- valence adjustment ---------------------------------------------------------------------


def test_valence_adjustment_hand_values() -> None:
    batch = _build_batch((_SULFONE,))
    model = _build_model(batch, **_REWIRE_KWARGS)
    atoms = th.arange(2, 14)  # the two phenyls without S, O, O (atoms 0 O, 1 S, 2 O? see assert)
    numbers = batch.atom_atomic_numbers
    # Product of the extrusion: the 12 carbons (S and both O deleted).
    atoms = th.nonzero(numbers == 6).flatten()
    assert atoms.numel() == 12
    group = th.zeros(12, dtype=th.long)
    counts = th.zeros((1, model.required_atomic_numbers.numel()), dtype=th.long)
    hydrogen_column = model.hydrogen_count_column
    counts[0, hydrogen_column] = -10  # complement hydrogens become molecule hydrogens + 10 = 20
    shifts = th.arange(0, 20)

    def allowed(adjustment):
        mask = model._transferable_hydrogen_shifts(
            batch,
            atoms,
            group,
            th.zeros(1, dtype=th.long),
            counts,
            shifts,
            th.tensor([2]),
            numbers,
            batch.atom_hydrogen_counts,
            1,
            None,
            None if adjustment is None else th.tensor([adjustment]),
        )
        return int(shifts[mask[0]].max())

    slots = int(batch.atom_free_valence_slots.long()[atoms].sum())
    assert slots == 12  # every aromatic carbon contributes one slot (correction C2)
    assert allowed(None) == slots + 2 == 14
    assert allowed(-2) == 12
    assert allowed(2) == 16
    assert allowed(-30) == 0  # negative gain clamps to the zero shift: shift 0 allowed only
    mask = model._transferable_hydrogen_shifts(
        batch, atoms, group, th.zeros(1, dtype=th.long), counts, shifts, th.tensor([2]),
        numbers, batch.atom_hydrogen_counts, 1, None, th.tensor([-30]),
    )
    assert mask[0].sum() == 1 and mask[0, 0]


# --- checkpoint ------------------------------------------------------------------------------


def test_checkpoint_roundtrip_and_prefix_loading() -> None:
    batch, _, on = _pair((_SULFONE,))
    on.intensity_head.rewire_pathway_residual.weight.data.normal_()
    fresh = _build_batch((_SULFONE,))
    other = _build_model(fresh, **_REWIRE_KWARGS)
    other.load_state_dict(on.state_dict())
    assert th.equal(
        other.intensity_head.rewire_pathway_residual.weight,
        on.intensity_head.rewire_pathway_residual.weight,
    )

    # A checkpoint that knows only the first n rules: later columns are zero-padded.
    n = MASSTER_NUM_REWIRE_RULES - 3
    state = copy.deepcopy(on.state_dict())
    state["rewire_rule_name_hashes"] = state["rewire_rule_name_hashes"][:n]
    state["intensity_head.rewire_pathway_residual.weight"] = state[
        "intensity_head.rewire_pathway_residual.weight"
    ][:, :n]
    prefix_model = _build_model(fresh, **_REWIRE_KWARGS)
    prefix_model.load_state_dict(state)
    assert prefix_model.rewire_rule_name_hashes.numel() == n
    weight = prefix_model.intensity_head.rewire_pathway_residual.weight
    assert weight.shape[1] == MASSTER_NUM_REWIRE_RULES
    assert th.equal(weight[:, :n], on.intensity_head.rewire_pathway_residual.weight[:, :n])
    assert not weight[:, n:].any()

    # Rules at or beyond the prefix produce no extras for that model.
    batch2 = _build_batch((_METHYL_SULFONE,))
    out = _run(prefix_model, batch2, (_METHYL_SULFONE,))
    assert out["rewire_is_extra"].any()
    short = copy.deepcopy(state)
    short["rewire_rule_name_hashes"] = short["rewire_rule_name_hashes"][:0]
    short["intensity_head.rewire_pathway_residual.weight"] = short[
        "intensity_head.rewire_pathway_residual.weight"
    ][:, :0]
    none_model = _build_model(fresh, **_REWIRE_KWARGS)
    none_model.load_state_dict(short)
    assert not _run(none_model, batch2, (_METHYL_SULFONE,))["rewire_is_extra"].any()


@pytest.mark.parametrize("change", ["reorder", "rename", "extra"])
def test_checkpoint_with_changed_rule_hashes_is_rejected(change: str) -> None:
    batch, _, on = _pair((_SULFONE,))
    state = copy.deepcopy(on.state_dict())
    hashes = state["rewire_rule_name_hashes"]
    if change == "reorder":
        hashes[[0, 1]] = hashes[[1, 0]]
    elif change == "rename":
        hashes[3] += 1
    else:
        state["rewire_rule_name_hashes"] = th.cat([hashes, hashes[:1]])
    target = _build_model(batch, **_REWIRE_KWARGS)
    with pytest.raises(ValueError, match="prefix of REWIRE_RULE_VOCABULARY"):
        target.load_state_dict(state)


def test_control_checkpoint_loads_into_on_model_as_the_control() -> None:
    batch = _build_batch((_SULFONE,))
    off = _build_model(batch, **_OFF_KWARGS)
    on = _build_model(batch, **_REWIRE_KWARGS)
    hashes = on.rewire_rule_name_hashes.clone()
    result = on.load_state_dict(off.state_dict())
    assert not result.missing_keys and not result.unexpected_keys
    assert th.equal(on.rewire_rule_name_hashes, hashes)
    assert not on.intensity_head.rewire_pathway_residual.weight.any()
    for key, value in off.state_dict().items():
        assert th.equal(value, on.state_dict()[key]), key


def test_residual_without_rule_list_is_rejected_when_on() -> None:
    batch, off, on = _pair((_SULFONE,))
    state = on.state_dict()
    del state["rewire_rule_name_hashes"]
    with pytest.raises(ValueError, match="no rewire rule list"):
        on.load_state_dict(state)


def test_flag_off_model_rejects_a_flag_on_checkpoint() -> None:
    batch, off, on = _pair((_SULFONE,))
    with pytest.raises(RuntimeError, match="rewire"):
        off.load_state_dict(on.state_dict())


# --- coexistence and consumers --------------------------------------------------------------


def test_smarts_and_rewire_coexist() -> None:
    smiles = (_SULFONAMIDE,)
    batch = _build_batch(smiles)
    kwargs = {**_REWIRE_KWARGS, "use_smarts_templates": True}
    model = _build_model(batch, **kwargs)
    out = _run(model, batch, smiles, smarts=True, return_annotation=True)
    assert out["rewire_is_extra"].any()
    assert th.isfinite(out["pred_logprobs"]).all()
    # SMARTS indicators of the appended rows are zero and indices do not collide.
    n_rows = out["ion_h_fragment_batch_idxs"].numel()
    assert out["fragment_rule_fragment_idxs"].max() < n_rows
    extra_rows = set(th.nonzero(out["rewire_is_extra"]).flatten().tolist())
    assert not extra_rows & set(out["fragment_rule_fragment_idxs"].tolist())


def test_share_proposal_across_ev_shares_extras() -> None:
    smiles = (_SULFONE, _SULFONE)
    batch = _build_batch(smiles)
    kwargs = {**_REWIRE_KWARGS, "share_proposal_across_ev": True}
    model = _build_model(batch, **kwargs)
    out = model(
        mol_pyg=batch,
        spec_prec_type=th.zeros(2, dtype=th.long),
        spec_nce=th.tensor([20.0, 40.0]),
        static_graph=_static_graph(batch, smiles),
        proposal_group_ids=th.tensor([0, 0]),
        return_annotation=True,
    )
    spectrum = out["ion_h_fragment_batch_idxs"]
    extra = out["rewire_is_extra"]
    sets = []
    for index in range(2):
        rows = th.nonzero(extra & (spectrum == index)).flatten()
        members = out["fragment_atom_indices"]
        fragments = out["fragment_atom_fragment_idxs"]
        sets.append(
            sorted(
                tuple(sorted((members[fragments == row] - batch.ptr[index]).tolist()))
                for row in rows.tolist()
            )
        )
    assert sets[0] and sets[0] == sets[1]


def test_memorization_sink_indexes_extras_by_their_own_spectrum() -> None:
    smiles = (_SULFONE, "CCO")
    batch = _build_batch(smiles)
    model = _build_model(
        batch, **_REWIRE_KWARGS, memorization_sink_dim=4, memorization_sink_num_molecules=3
    )
    out = _run(model, batch, smiles, memorization_sink_indices=th.tensor([0, 1]))
    assert out["rewire_is_extra"].any() and th.isfinite(out["pred_logprobs"]).all()


def test_full_loss_and_metrics_are_finite_on_extended_tensors() -> None:
    smiles = (_SULFONE, _METHYL_SULFONE)
    batch = _build_batch(smiles)
    kwargs = {**_REWIRE_KWARGS, "separate_ion_h_support_head": True, "ion_h_support_binary_logits": True}
    model = _build_model(batch, **kwargs)
    with th.autograd.set_detect_anomaly(True):
        out = _run(model, batch, smiles)
        true_mzs = th.tensor([100.0, 154.0, 77.0, 150.0])
        true_batch = th.tensor([0, 0, 1, 1])
        true_logprobs = th.log(th.full((4,), 0.5))
        coverage = _proposal_coverage_loss(
            true_mzs=true_mzs,
            true_logprobs=true_logprobs,
            true_batch_idxs=true_batch,
            candidate_mzs=out["proposal_candidate_mzs"],
            candidate_fragment_idxs=out["proposal_candidate_fragment_idxs"],
            fragment_logprobs=out["proposal_fragment_logprobs"],
            fragment_batch_idxs=out["proposal_fragment_batch_idxs"],
            batch_size=2,
            tolerance=0.01,
            relative=False,
            tolerance_min_mz=0.0,
            tol_per_sample=None,
            min_mz_per_sample=None,
            barcode_target="pool",
        )
        channels = out["ion_h_chemical_valid_channels"]
        targets = th.rand(channels.shape) * channels
        metrics = _channel_support_metrics(
            channel_targets=targets,
            chemical_valid_channels=channels,
            fragment_batch_idxs=out["ion_h_fragment_batch_idxs"],
            batch_size=2,
            fragment_logprobs=out["ion_h_fragment_proposal_logprobs"],
        )
        loss = (
            -(out["pred_logprobs"] * th.linspace(0.2, 1.0, out["pred_logprobs"].numel())).sum()
            + coverage.sum()
            + (out["ion_h_support_logits"] * targets).sum()
        )
        loss.backward()
    assert all(th.isfinite(value).all() for value in metrics.values())
    for name, parameter in model.named_parameters():
        assert parameter.grad is None or th.isfinite(parameter.grad).all(), name
    assert math.isfinite(float(loss))


# --- CPU / CUDA ------------------------------------------------------------------------------


@pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA is not available")
def test_cpu_and_cuda_backends_give_the_same_extras() -> None:
    smiles = (_SULFONE, _METHYL_SULFONE, "CS(=O)(=O)C", _SULFONE)
    cpu_batch = _build_batch(smiles)
    cuda_batch = cpu_batch.clone().cuda()
    static = _static_graph(cpu_batch, smiles)
    inputs = dict(
        mol_pyg=cuda_batch,
        spec_prec_type=th.zeros(len(smiles), dtype=th.long, device="cuda"),
        spec_nce=th.full((len(smiles),), 30.0, device="cuda"),
        static_graph=static,
        proposal_group_ids=th.tensor([0, 1, 2, 0], device="cuda"),
        return_annotation=True,
    )
    kwargs = {**_REWIRE_KWARGS, "share_proposal_across_ev": True}
    results = {}
    reference = None
    for backend in ("cpu", "cuda"):
        model = _build_model(cpu_batch, **kwargs, candidate_backend=backend).cuda().eval()
        if reference is not None:
            model.load_state_dict(reference.state_dict())
        reference = reference or model
        out = model(**inputs)
        members = out["fragment_atom_indices"].cpu()
        fragments = out["fragment_atom_fragment_idxs"].cpu()
        batches = out["ion_h_fragment_batch_idxs"].cpu()
        results[backend] = {
            "is_extra": out["rewire_is_extra"].cpu(),
            "sets": [
                (int(batches[row]), tuple(sorted(members[fragments == row].tolist())))
                for row in range(batches.numel())
            ],
            "parents": out["fragment_rewire_parent_idxs"].cpu(),
            "pred_mzs": out["pred_mzs"].cpu(),
            "pred_logprobs": out["pred_logprobs"].cpu(),
        }
    cpu, cuda = results["cpu"], results["cuda"]
    assert cpu["sets"] == cuda["sets"]  # same unique-fragment order, keep mask and extra order
    assert _same(cpu["is_extra"], cuda["is_extra"])
    assert _same(cpu["parents"], cuda["parents"])
    th.testing.assert_close(cpu["pred_mzs"], cuda["pred_mzs"], atol=1e-6, rtol=1e-5)
    th.testing.assert_close(cpu["pred_logprobs"], cuda["pred_logprobs"], atol=1e-5, rtol=1e-4)
