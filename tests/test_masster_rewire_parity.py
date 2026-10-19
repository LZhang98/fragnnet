"""Cross-cutting parity tests of the MASSter rewire stage (build step 6, arm a0)."""

import copy

import pytest
import torch as th

from tests.test_masster_model import _build_batch, _build_model
from tests.test_masster_rewire_model import (
    _OFF_KWARGS,
    _REWIRE_KWARGS,
    _METHYL_SULFONE,
    _SULFONE,
    _SULFONAMIDE,
    _pair,
    _run,
    _same,
    _static_graph,
)

# Eight diverse molecules no admitted rule matches (checked by the first test).
_NO_MATCH = (
    "CCO",
    "CC(=O)Nc1ccc(O)cc1",
    "c1ccccc1",
    "CCN(CC)CC",
    "OC(=O)CCCC(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "C1CCOC1",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
)
_PROPOSAL_KEYS = (
    "proposal_candidate_mzs",
    "proposal_candidate_fragment_idxs",
    "proposal_fragment_logprobs",
    "proposal_fragment_batch_idxs",
    "bond_break_logits",
    "fragment_count_before_cutoff",
    "retained_fragment_probability",
)
_SKIP = ("timing", "_ms")


def _tensors(out):
    return {
        key: value
        for key, value in out.items()
        if isinstance(value, th.Tensor) and not any(token in key for token in _SKIP)
    }


def test_no_match_molecules_bit_identical_through_the_full_model() -> None:
    batch, off, on = _pair(_NO_MATCH)
    off.eval()
    on.eval()
    with th.no_grad():
        control = _run(off, batch, _NO_MATCH, return_annotation=True)
        out = _run(on, batch, _NO_MATCH, return_annotation=True)
    assert out["rewire_match_count"].tolist() == [0.0] * len(_NO_MATCH)
    assert not out["rewire_is_extra"].any()
    assert not [key for key in control if key not in out]
    for key, value in _tensors(control).items():
        assert _same(value, out[key]), key
    added = set(_tensors(out)) - set(_tensors(control))
    assert added and all("rewire" in key for key in added), added


def test_no_match_gradients_equal_the_control() -> None:
    smiles = _NO_MATCH[:3]
    batch, off, on = _pair(smiles)
    grads = []
    for model in (off, on):
        out = _run(model, batch, smiles)
        loss = (out["pred_logprobs"] * th.linspace(0.5, 1.5, out["pred_logprobs"].numel())).sum()
        params = [p for name, p in model.named_parameters() if "rewire" not in name]
        grads.append(th.autograd.grad(loss, params, allow_unused=True))
    for left, right in zip(*grads):
        assert (left is None) == (right is None)
        if left is not None:
            assert _same(left, right)


@pytest.mark.parametrize(
    "smiles",
    [
        (_SULFONE, _METHYL_SULFONE, _SULFONAMIDE, "CCO", "O=S1(=O)c2ccccc2-c2ccccc12"),
        (_SULFONE,) + _NO_MATCH[:3],
    ],
)
def test_proposal_outputs_equal_the_control_for_every_molecule(smiles) -> None:
    batch, off, on = _pair(smiles)
    off.eval()
    on.eval()
    with th.no_grad():
        control = _run(off, batch, smiles)
        out = _run(on, batch, smiles)
    assert out["rewire_is_extra"].any()
    for key in _PROPOSAL_KEYS:
        assert _same(control[key], out[key]), key
    ordinary = ~out["rewire_is_extra"]
    # Ordinary rows keep the control's order, batch ids and proposal log-probabilities.
    assert _same(
        control["ion_h_fragment_batch_idxs"], out["ion_h_fragment_batch_idxs"][ordinary]
    )
    assert _same(
        control["ion_h_fragment_proposal_logprobs"],
        out["ion_h_fragment_proposal_logprobs"][ordinary],
    )


def _split(out, num_spectra):
    """Per-spectrum views keyed by the spectrum index of the run."""
    rows = out["ion_h_fragment_batch_idxs"]
    result = {}
    for index in range(num_spectra):
        pred = out["pred_batch_idxs"] == index
        mask = rows == index
        result[index] = dict(
            is_extra=out["rewire_is_extra"][mask],
            pred_mzs=out["pred_mzs"][pred],
            pred_logprobs=out["pred_logprobs"][pred],
            valid=out["ion_h_chemical_valid_channels"][mask],
            proposal=out["ion_h_fragment_proposal_logprobs"][mask],
        )
    return result


@pytest.mark.parametrize("order", [[2, 0, 3, 1], [3, 2, 1, 0]])
def test_shuffling_the_spectra_changes_nothing_per_spectrum(order) -> None:
    smiles = (_SULFONE, "CCO", _METHYL_SULFONE, _SULFONAMIDE)
    batch, _, on = _pair(smiles)
    on.eval()
    shuffled_smiles = tuple(smiles[i] for i in order)
    shuffled = _build_batch(shuffled_smiles)
    with th.no_grad():
        base = _split(_run(on, batch, smiles), len(smiles))
        moved = _split(_run(on, shuffled, shuffled_smiles), len(smiles))
    for new_index, old_index in enumerate(order):
        a, b = base[old_index], moved[new_index]
        for key in a:
            if key in ("pred_logprobs", "proposal"):
                th.testing.assert_close(a[key], b[key], atol=1e-6, rtol=1e-5)
            else:
                assert _same(a[key], b[key]), (key, old_index)


@pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA is not available")
def test_shuffled_spectra_on_cuda_match_the_cpu_per_spectrum() -> None:
    smiles = (_SULFONE, "CCO", _METHYL_SULFONE)
    order = [2, 0, 1]
    shuffled_smiles = tuple(smiles[i] for i in order)
    cpu_batch = _build_batch(shuffled_smiles)
    model = _build_model(cpu_batch, **_REWIRE_KWARGS).eval()
    with th.no_grad():
        cpu = _split(_run(model, cpu_batch, shuffled_smiles), 3)
        cuda_model = _build_model(cpu_batch, **_REWIRE_KWARGS).cuda().eval()
        cuda_model.load_state_dict(model.state_dict())
        cuda_batch = cpu_batch.clone().cuda()
        out = cuda_model(
            mol_pyg=cuda_batch,
            spec_prec_type=th.zeros(3, dtype=th.long, device="cuda"),
            spec_nce=th.full((3,), 30.0, device="cuda"),
            static_graph=_static_graph(cpu_batch, shuffled_smiles),
        )
        cuda = _split({k: v.cpu() for k, v in _tensors(out).items()}, 3)
    for index in range(3):
        assert _same(cpu[index]["is_extra"], cuda[index]["is_extra"])
        th.testing.assert_close(cpu[index]["pred_mzs"], cuda[index]["pred_mzs"], atol=1e-6, rtol=1e-5)


# --- checkpoint with and without the flag -----------------------------------------------------


def test_off_checkpoint_has_no_rewire_keys_and_loads_unchanged() -> None:
    batch = _build_batch((_SULFONE,))
    off = _build_model(batch, **_OFF_KWARGS).eval()
    state = off.state_dict()
    assert not [key for key in state if "rewire" in key]
    twin = _build_model(_build_batch((_SULFONE,)), **_OFF_KWARGS).eval()
    result = twin.load_state_dict(state)
    assert not result.missing_keys and not result.unexpected_keys
    with th.no_grad():
        assert _same(
            _run(off, batch, (_SULFONE,))["pred_logprobs"],
            _run(twin, batch, (_SULFONE,))["pred_logprobs"],
        )


def test_on_checkpoint_round_trips_and_an_off_model_matches_its_shared_part() -> None:
    batch, off, on = _pair((_SULFONE,))
    on.eval()
    on_state = on.state_dict()
    rewire_keys = {key for key in on_state if "rewire" in key}
    assert rewire_keys
    off_state = off.state_dict()
    # The flag-on state dict is the OFF one plus rewire entries only.
    assert set(on_state) - set(off_state) == rewire_keys
    for key in off_state:
        assert _same(off_state[key], on_state[key]), key
    clone = _build_model(_build_batch((_SULFONE,)), **_REWIRE_KWARGS).eval()
    clone.load_state_dict(copy.deepcopy(on_state))
    with th.no_grad():
        a = _run(on, batch, (_SULFONE,))
        b = _run(clone, batch, (_SULFONE,))
    for key in ("pred_mzs", "pred_logprobs", "rewire_is_extra"):
        assert _same(a[key], b[key]), key


# --- hand-computed valence / hydrogen-channel cases -------------------------------------------


def test_so2_extrusion_extra_hydrogen_channels_follow_the_adjusted_valence_bound() -> None:
    """Diphenyl sulfone -> biphenyl-like 12-carbon product: valence_gain 12 (14 without the -2)."""
    smiles = (_SULFONE,)
    batch, _, on = _pair(smiles)
    on.eval()
    with th.no_grad():
        out = _run(on, batch, smiles, return_annotation=True)
    is_extra = out["rewire_is_extra"]
    sets = out["fragment_atom_indices"]
    fragments = out["fragment_atom_fragment_idxs"]
    numbers = batch.atom_atomic_numbers
    target = None
    for row in th.nonzero(is_extra).flatten().tolist():
        members = sets[fragments == row]
        if members.numel() == 12 and bool((numbers[members] == 6).all()):
            target = row
    assert target is not None, "the 12-carbon extrusion product is an extra"
    valid = out["ion_h_chemical_valid_channels"][target]
    assert valid.any()
    shifts = out["pred_pathway_h_shifts"] if out["pred_pathway_h_shifts"].numel() else None
    # Every channel the model keeps lies at or below the adjusted bound of 12 shifts: the largest
    # allowed shift of the stage's own bound call (adjustment -2) is what the valid mask encodes.
    bound = _bound(on, batch, members_of(sets, fragments, target), adjustment=-2)
    unadjusted = _bound(on, batch, members_of(sets, fragments, target), adjustment=None)
    assert (bound, unadjusted) == (12, 14)
    assert shifts is None or int(shifts.max()) <= 14


def members_of(sets, fragments, row):
    return sets[fragments == row]


def _bound(model, batch, atoms, adjustment):
    numbers = batch.atom_atomic_numbers
    counts = th.zeros((1, model.required_atomic_numbers.numel()), dtype=th.long)
    counts[0, model.hydrogen_count_column] = -10
    shifts = th.arange(0, 20)
    mask = model._transferable_hydrogen_shifts(
        batch,
        atoms,
        th.zeros(atoms.numel(), dtype=th.long),
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


# --- CPU / CUDA stage parity ------------------------------------------------------------------


@pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA is not available")
def test_stage_valid_channels_agree_between_cpu_and_cuda_backends() -> None:
    smiles = (_SULFONE, _METHYL_SULFONE, _SULFONAMIDE)
    cpu_batch = _build_batch(smiles)
    cpu = _build_model(cpu_batch, **_REWIRE_KWARGS, candidate_backend="cpu").cuda().eval()
    cuda = _build_model(cpu_batch, **_REWIRE_KWARGS, candidate_backend="cuda").cuda().eval()
    cuda.load_state_dict(cpu.state_dict())
    outs = []
    for model in (cpu, cuda):
        with th.no_grad():
            out = model(
                mol_pyg=cpu_batch.clone().cuda(),
                spec_prec_type=th.zeros(3, dtype=th.long, device="cuda"),
                spec_nce=th.full((3,), 30.0, device="cuda"),
                static_graph=_static_graph(cpu_batch, smiles),
            )
        outs.append({k: v.cpu() for k, v in _tensors(out).items()})
    assert _same(outs[0]["rewire_is_extra"], outs[1]["rewire_is_extra"])
    assert _same(
        outs[0]["ion_h_chemical_valid_channels"], outs[1]["ion_h_chemical_valid_channels"]
    )
