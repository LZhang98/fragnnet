"""MASSter nominal-mass channels and natural isotope rows for unit-resolution EI.

Plan: docs/plans/masster/masster_ei_nominal_plan.md.
"""

from collections import defaultdict

import numpy as np
import pytest
import torch as th
from rdkit import Chem

from fragnnet.model.candidate_features import COMPOSITION_FEATURE_DIM
from fragnnet.utils.formula_utils import nominal_isotope_power_table
from fragnnet.utils.frag_utils import CANONICAL_ELEMENT_ORDER
from tests.test_masster_model import _build_batch, _build_model

NOMINAL = {"mass_mode": "nominal"}
NATURAL = {"mass_mode": "nominal", "isotope_envelope": "natural"}


def _reference_pattern(counts: dict[str, int], order: int, min_ratio: float) -> np.ndarray:
    """Independent full convolution of RDKit abundances, cropped only at the end."""
    periodic_table = Chem.GetPeriodicTable()
    pattern = {0: 1.0}
    for element, count in counts.items():
        base = periodic_table.GetMostCommonIsotope(element)
        single = {
            mass - base: periodic_table.GetAbundanceForIsotope(element, mass) / 100.0
            for mass in range(1, 300)
            if periodic_table.GetAbundanceForIsotope(element, mass) > 0
        }
        for _ in range(count):
            combined = defaultdict(float)
            for offset, probability in pattern.items():
                for delta, abundance in single.items():
                    combined[offset + delta] += probability * abundance
            pattern = combined
    result = np.array([pattern.get(offset, 0.0) for offset in range(-order, order + 1)])
    result /= result.sum()
    result[result < min_ratio] = 0.0
    return result / result.sum()


def test_nominal_isotope_power_table_matches_natural_abundance() -> None:
    chlorine = nominal_isotope_power_table("Cl", 1, 2)[1]
    assert chlorine == pytest.approx([0.0, 0.0, 0.7576, 0.0, 0.2424], abs=2e-3)
    dibromo = nominal_isotope_power_table("Br", 2, 4)[2]
    assert dibromo[[4, 6, 8]] == pytest.approx([0.2569, 0.4999, 0.2432], abs=2e-3)
    assert dibromo.sum() == pytest.approx(1.0)
    fluorine = nominal_isotope_power_table("F", 7, 4)
    assert (fluorine[:, 4] == 1.0).all()
    # 78Se sits two units below 80Se, the most abundant isotope.
    selenium = nominal_isotope_power_table("Se", 1, 4)[1]
    assert selenium[4] == selenium.max()
    assert selenium[2] > 0.2


@pytest.mark.parametrize(
    ("smiles", "atom_indices", "adduct", "nominal"),
    [
        ("CC(F)(F)F", [1, 2, 3, 4], "[M]+", 69),  # CF3+, exact 68.995: floor binning puts it at 68
        ("ClC(Cl)(Cl)Cl", [1, 2, 3, 4], "[M]+", 117),  # CCl3+, exact 116.906
        ("CCO", [0, 1, 2], "[M+H]+", 47),
        ("CCO", [0, 1, 2], "[M-H]-", 45),
    ],
)
def test_nominal_channels_are_mass_number_sums(
    smiles: str, atom_indices: list[int], adduct: str, nominal: int
) -> None:
    batch = _build_batch((smiles,))
    model = _build_model(
        batch, precursor_types=(adduct,), include_cmf=False, max_h_transfer=1, **NOMINAL
    )
    masses, valid = model._fragment_masses_and_valid_channels(
        batch,
        th.tensor(atom_indices),
        th.zeros(len(atom_indices), dtype=th.long),
        th.tensor([0]),
        1,
        th.tensor([0]),
    )
    assert masses[0, 0].tolist() == [nominal - 1, nominal, nominal + 1]
    assert valid[0, 0, 1]


def test_isotope_rows_match_reference_convolution() -> None:
    batch = _build_batch(("C[Se]C",))
    model = _build_model(batch, precursor_types=("[M]+",), include_cmf=False, **NATURAL)
    counts = th.zeros((1, COMPOSITION_FEATURE_DIM), dtype=th.long)
    for element, count in {"C": 2, "H": 6, "Se": 1, "Br": 1}.items():
        counts[0, CANONICAL_ELEMENT_ORDER.index(element)] = count
    shifts = th.tensor([0, -1, 2])
    rows, offsets, log_shares = model._isotope_rows(
        counts, th.tensor([0]), th.zeros(3, dtype=th.long), th.zeros(3, dtype=th.long), shifts
    )
    order = model.isotope_max_order
    for row, shift in enumerate(shifts.tolist()):
        expected = _reference_pattern(
            {"C": 2, "H": 6 + shift, "Se": 1, "Br": 1}, order, model.isotope_min_ratio
        )
        observed = np.zeros(2 * order + 1)
        mine = rows == row
        observed[(offsets[mine] + order).numpy()] = log_shares[mine].exp().double().numpy()
        assert observed == pytest.approx(expected, abs=1e-6)
    assert (offsets < 0).any()  # the selenium isotopes below 80Se


def test_natural_isotope_rows_split_each_pathway() -> None:
    batch = _build_batch(("CCBr", "ClCCO"))
    inputs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 0]),
        "spec_nce": th.tensor([20.0, 40.0]),
        "return_annotation": True,
    }
    common = {"precursor_types": ("[M]+",), "include_cmf": False, "max_h_transfer": 2}
    mono = _build_model(batch, **common, **NOMINAL).eval()(**inputs)
    isotopic = _build_model(batch, **common, **NATURAL).eval()(**inputs)

    keys = ("pred_pathway_fragment_idxs", "pred_pathway_ion_mode_idxs", "pred_pathway_h_shifts")
    mono_rows = {tuple(int(mono[k][i]) for k in keys): i for i in range(mono["pred_mzs"].numel())}
    assert "pred_pathway_isotope_offsets" not in mono
    offsets = isotopic["pred_pathway_isotope_offsets"]
    assert offsets.shape == isotopic["pred_mzs"].shape
    split = {}
    for i in range(isotopic["pred_mzs"].numel()):
        source = mono_rows[tuple(int(isotopic[k][i]) for k in keys)]
        assert isotopic["pred_mzs"][i] == mono["pred_mzs"][source] + offsets[i]
        assert isotopic["pred_batch_idxs"][i] == mono["pred_batch_idxs"][source]
        split[source] = split.get(source, 0.0) + float(isotopic["pred_logprobs"][i].exp())
    assert sorted(split) == list(range(mono["pred_mzs"].numel()))
    for source, total in split.items():
        assert total == pytest.approx(float(mono["pred_logprobs"][source].exp()), rel=1e-5)
    # Bromine and chlorine put a real M+2 on their fragments.
    assert (offsets == 2).any()
    assert th.equal(mono["pred_mzs"], mono["pred_mzs"].round())


@pytest.mark.parametrize("unique", [False, True])
def test_isotope_rows_follow_each_fragments_atoms(unique: bool) -> None:
    batch = _build_batch(("CCBr", "ClCC(=O)OC"))
    extra = {"unique_fragment_mode": True, "max_masks": 6, "max_enumerated_masks": 16}
    model = _build_model(
        batch,
        precursor_types=("[M]+",),
        include_cmf=False,
        max_h_transfer=2,
        **(extra if unique else {}),
        **NATURAL,
    ).eval()
    out = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0, 0]),
        spec_nce=th.tensor([20.0, 40.0]),
        return_annotation=True,
    )

    # Each fragment's composition, read from the annotation's own atom memberships.
    symbols = [Chem.GetPeriodicTable().GetElementSymbol(int(z)) for z in batch.atom_atomic_numbers]
    compositions: dict[int, dict[str, int]] = {}
    for atom, fragment in zip(
        out["fragment_atom_indices"].tolist(), out["fragment_atom_fragment_idxs"].tolist()
    ):
        counts = compositions.setdefault(fragment, {"H": 0})
        counts[symbols[atom]] = counts.get(symbols[atom], 0) + 1
        counts["H"] += int(batch.atom_hydrogen_counts[atom])
    order = model.isotope_max_order
    patterns: dict[tuple[int, int], np.ndarray] = {}
    for fragment, shift, offset, logprob in zip(
        out["pred_pathway_fragment_idxs"].tolist(),
        out["pred_pathway_h_shifts"].tolist(),
        out["pred_pathway_isotope_offsets"].tolist(),
        out["pred_logprobs"].exp().tolist(),
    ):
        patterns.setdefault((fragment, shift), np.zeros(2 * order + 1))[offset + order] = logprob
    for (fragment, shift), observed in patterns.items():
        counts = dict(compositions[fragment])
        counts["H"] += shift
        expected = _reference_pattern(counts, order, model.isotope_min_ratio)
        assert observed / observed.sum() == pytest.approx(expected, abs=1e-5)
    halogenated = [f for f, c in compositions.items() if "Br" in c or "Cl" in c]
    assert halogenated
    assert any(patterns[key][order + 2] > 0 for key in patterns if key[0] in halogenated)


@pytest.mark.parametrize("settings", [NOMINAL, NATURAL])
def test_nominal_rows_merge_by_spectrum_and_mz(settings: dict) -> None:
    batch = _build_batch(("CCBr", "ClCCO"))
    model = _build_model(
        batch, precursor_types=("[M]+",), include_cmf=False, max_h_transfer=2, **settings
    ).eval()
    inputs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 0]),
        "spec_nce": th.tensor([20.0, 40.0]),
    }
    merged = model(**inputs)
    rows = model(**inputs, return_annotation=True)

    expected = {}
    for mz, logprob, batch_idx in zip(
        rows["pred_mzs"].tolist(),
        rows["pred_logprobs"].exp().tolist(),
        rows["pred_batch_idxs"].tolist(),
    ):
        expected[(batch_idx, mz)] = expected.get((batch_idx, mz), 0.0) + logprob
    observed = {
        (int(b), float(mz)): float(p.exp())
        for mz, p, b in zip(merged["pred_mzs"], merged["pred_logprobs"], merged["pred_batch_idxs"])
    }
    assert observed.keys() == expected.keys()
    assert merged["pred_mzs"].numel() < rows["pred_mzs"].numel()
    for key, value in expected.items():
        assert observed[key] == pytest.approx(value, rel=1e-5)


def test_nominal_isotope_model_loads_exact_checkpoints_strict() -> None:
    batch = _build_batch()
    exact = _build_model(batch)
    isotopic = _build_model(batch, **NATURAL)
    isotopic.load_state_dict(exact.state_dict(), strict=True)
    exact.load_state_dict(isotopic.state_dict(), strict=True)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mass_mode": "binned"}, "mass_mode"),
        ({"isotope_envelope": "natural"}, "requires mass_mode='nominal'"),
        ({**NATURAL, "isotope_max_order": 0}, "isotope_max_order"),
        ({**NATURAL, "isotope_min_ratio": 0.2}, "isotope_min_ratio"),
    ],
)
def test_invalid_mass_and_isotope_settings_raise(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _build_model(_build_batch(), **kwargs)


@pytest.mark.parametrize("element,count", [("Se", 2), ("B", 4), ("Hg", 2), ("Sn", 2)])
def test_power_table_keeps_offsets_that_return_to_window(element: str, count: int) -> None:
    pattern = nominal_isotope_power_table(element, count, 4)[count]
    assert pattern / pattern.sum() == pytest.approx(
        _reference_pattern({element: count}, 4, 0.0), abs=1e-12
    )


@pytest.mark.parametrize("order", [1, 4, 6])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_isotope_rows_cancel_offsets_across_elements(order: int, device: str) -> None:
    if device == "cuda" and not th.cuda.is_available():
        pytest.skip("CUDA unavailable")
    compositions = [
        {"Se": 1, "Br": 3, "H": 2},
        {"Se": 2, "C": 2, "H": 6},
        {"B": 3, "Cl": 2, "H": 2},
        {"Hg": 1, "Br": 2, "H": 2},
        {"C": 2, "Cl": 1, "H": 6},
    ]
    batch = _build_batch(("C[Se]C",))
    model = _build_model(
        batch, precursor_types=("[M]+",), include_cmf=False,
        isotope_max_order=order, **NATURAL,
    ).to(device)
    counts = th.zeros((len(compositions), COMPOSITION_FEATURE_DIM), dtype=th.long, device=device)
    for i, composition in enumerate(compositions):
        for element, count in composition.items():
            counts[i, CANONICAL_ELEMENT_ORDER.index(element)] = count
    fragments = th.arange(len(compositions), device=device).repeat_interleave(3)
    shifts = th.tensor([-1, 0, 2], device=device).repeat(len(compositions))
    rows, offsets, shares = model._isotope_rows(
        counts, th.zeros(len(compositions), dtype=th.long, device=device),
        fragments, th.zeros_like(fragments), shifts,
    )
    for row, (fragment, shift) in enumerate(zip(fragments.tolist(), shifts.tolist())):
        composition = dict(compositions[fragment])
        composition["H"] += shift
        expected = _reference_pattern(composition, order, model.isotope_min_ratio)
        observed = np.zeros(2 * order + 1)
        selected = rows == row
        observed[(offsets[selected] + order).cpu().numpy()] = shares[selected].exp().cpu().numpy()
        assert observed == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_negative_offset_isotopes_forward_backward(device: str) -> None:
    if device == "cuda" and not th.cuda.is_available():
        pytest.skip("CUDA unavailable")
    batch = _build_batch(("C[Se]C", "CCBr")).to(device)
    model = _build_model(
        batch, precursor_types=("[M]+",), include_cmf=False, **NATURAL
    ).to(device).eval()
    inputs = {
        "mol_pyg": batch,
        "spec_prec_type": th.zeros(2, dtype=th.long, device=device),
        "spec_nce": th.tensor([20.0, 40.0], device=device),
    }
    out = model(**inputs)
    probabilities = out["pred_logprobs"].exp()
    totals = probabilities.new_zeros(2).index_add(0, out["pred_batch_idxs"], probabilities)
    assert th.allclose(totals, th.ones_like(totals), atol=1e-6)
    loss = -(probabilities * out["pred_logprobs"]).sum()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(th.isfinite(g).all() for g in gradients)
    assert any((g != 0).any() for g in gradients)
