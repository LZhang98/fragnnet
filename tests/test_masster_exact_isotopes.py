"""Exact isotope masses, fine structure, cancellation, and MASSter output conservation."""

from collections import defaultdict

import numpy as np
import pytest
import torch

from fragnnet.models.masster.masster_exact_isotopes import exact_isotope_pattern
from fragnnet.models.masster.masster_nominal_isotopes import nominal_isotope_pattern
from tests.test_masster_model import _build_batch, _build_model


def test_carbon_nitrogen_fine_structure_and_halogen_masses():
    pattern = exact_isotope_pattern((("C", 1), ("N", 1)), 4, 0)
    m1 = [(mass, probability) for offset, mass, probability in pattern if offset == 1]
    assert len(m1) == 2
    assert sorted(mass for mass, _ in m1) == pytest.approx([0.9970349, 1.00335484], abs=1e-7)
    chlorine = exact_isotope_pattern((("Cl", 1),), 4, 0)
    assert chlorine[1][1] == pytest.approx(1.99704992, abs=1e-7)
    assert sum(row[2] for row in pattern) == pytest.approx(1)


@pytest.mark.parametrize("composition", [(("Se", 2), ("Br", 2)), (("C", 5), ("H", 12), ("Si", 2))])
def test_fine_structure_sums_to_full_nominal_convolution(composition):
    pattern = exact_isotope_pattern(composition, 4, 0)
    grouped = defaultdict(float)
    for offset, _, probability in pattern:
        grouped[offset] += probability
    nominal = np.array(nominal_isotope_pattern(composition, 4))
    nominal /= nominal.sum()
    assert [grouped[offset] for offset in range(-4, 5)] == pytest.approx(nominal, abs=1e-10)


@pytest.mark.parametrize("adduct", ["[M]+", "[M+H]+", "[M-H]-"])
def test_exact_isotope_forward_conserves_scores_and_gradients(adduct):
    batch = _build_batch(("CCBr", "ClCCO"))
    common = {"precursor_types": (adduct,), "include_cmf": False, "max_h_transfer": 2}
    mono = _build_model(batch, **common).eval()
    model = _build_model(batch, isotope_envelope="natural", **common).eval()
    model.load_state_dict(mono.state_dict(), strict=True)
    inputs = {
        "mol_pyg": batch,
        "spec_prec_type": torch.zeros(2, dtype=torch.long),
        "spec_nce": torch.tensor([20.0, 40.0]),
        "return_annotation": True,
    }
    base = mono(**inputs)
    out = model(**inputs)
    # Exact mode with no merge retains one row per ion pathway in base prediction outputs.
    identities = [
        "pred_pathway_fragment_idxs",
        "pred_pathway_ion_mode_idxs",
        "pred_pathway_h_shifts",
    ]
    keys = [tuple(int(base[k][i]) for k in identities) for i in range(len(base["pred_mzs"]))]
    grouped = defaultdict(float)
    for i, (mz, score) in enumerate(zip(out["pred_mzs"], out["pred_logprobs"].exp())):
        key = tuple(int(out[k][i]) for k in identities)
        index = keys.index(key)
        assert float(mz) == pytest.approx(
            float(base["pred_mzs"][index]) + float(out["pred_pathway_isotope_mass_shifts"][i]),
            abs=1e-9,
        )
        grouped[key] += float(score.detach())
    for index, key in enumerate(keys):
        assert grouped[key] == pytest.approx(float(base["pred_logprobs"][index].exp()), rel=1e-5)
    totals = torch.zeros(2).index_add(0, out["pred_batch_idxs"], out["pred_logprobs"].exp())
    assert totals.tolist() == pytest.approx([1, 1], abs=1e-6)
    loss = -(out["pred_logprobs"].exp() * out["pred_logprobs"]).sum()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any((g != 0).any() for g in gradients)


def test_exact_isotope_plain_export_and_annotation_guard():
    from tests.test_inference_masster_peak_aggregation import _export, _MassterStub

    mz = torch.tensor([100.0, 100.9970349, 101.00335484], dtype=torch.float64)
    intensity = torch.tensor([0.97, 0.01, 0.02])
    batch_ids = torch.zeros(3, dtype=torch.long)
    [(exported_mz, exported_int)] = _export(mz, intensity, batch_ids)
    assert exported_mz == pytest.approx(mz.numpy(), abs=1e-7)
    assert exported_int == pytest.approx([1000, 1000 / 97, 2000 / 97], rel=1e-5)
    stub = _MassterStub({})
    with pytest.raises(NotImplementedError, match="does not label isotope rows"):
        stub.postprocess_export_peaks(
            {},
            {"pred_pathway_isotope_offsets": torch.tensor([0, 1, 1])},
            (mz, intensity, batch_ids),
            need_annotation=True,
            return_cc_hash=False,
            return_rule_provenance=False,
        )


def test_exact_isotopes_reach_real_sparse_training_loss():
    from tests.test_masster_rewire_pl_export import _batch, _pl

    pl = _pl(
        masster_mass_mode="exact",
        masster_isotope_envelope="natural",
        masster_rewire_rules=False,
        masster_include_cmf=False,
    ).eval()
    batch = _batch(pl, ("CCBr", "ClCCO"))
    out = pl.forward(**batch)
    masses = torch.stack([out["pred_mzs"][out["pred_batch_idxs"] == i][0] for i in [0, 1]])
    data = dict(out)
    data.update(
        true_mzs=masses,
        true_logprobs=torch.zeros(2),
        true_intensities=torch.ones(2),
        true_batch_idxs=torch.tensor([0, 1]),
        batch_size=2,
    )
    result = pl.loss_fn(**data)
    assert torch.isfinite(result["loss"]).all()
    result["loss"].sum().backward()
    gradients = [p.grad for p in pl.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


def test_exact_isotope_independent_intensities_conserve_each_spectrum():
    batch = _build_batch(("CCBr", "CCO"))
    options = {
        "unique_fragment_mode": True,
        "use_base_probability_feature": False,
        "proposal_selection_only": True,
        "intensity_output": "independent",
        "precursor_types": ("[M]+",),
        "include_cmf": False,
    }
    mono = _build_model(batch, **options).eval()
    isotopic = _build_model(batch, isotope_envelope="natural", **options).eval()
    isotopic.load_state_dict(mono.state_dict(), strict=True)
    inputs = {
        "mol_pyg": batch,
        "spec_prec_type": torch.zeros(2, dtype=torch.long),
        "spec_nce": torch.tensor([20.0, 40.0]),
    }
    base, out = mono(**inputs), isotopic(**inputs)
    totals = torch.zeros(2).index_add(0, out["pred_batch_idxs"], out["pred_intensities"])
    expected = torch.zeros(2).index_add(0, base["pred_batch_idxs"], base["pred_intensities"])
    assert torch.allclose(totals, expected, rtol=1e-5)
    totals.sum().backward()
    assert any(p.grad is not None and (p.grad != 0).any() for p in isotopic.parameters())
