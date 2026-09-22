"""Boundary-bond encoding and the mass/charge candidate scalars.

Covers the fixes recorded in ``docs/reference/masshatter_feature_set_audit.md`` §7.2, §7.4,
§7.5 and §7.7: the degenerate endpoint-average boundary embedding, the mass-blind candidate
token, the constant token columns in unique-fragment mode, and the unread required graph field.
"""

import pytest
import torch as th
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from fragnnet.model.base_model import CutBondEncoder
from fragnnet.model.bond_fragment_heads import (
    CandidateIntensityTransformerHead,
    boundary_bond_endpoint_orientation,
    pool_boundary_bond_embeddings,
)
from fragnnet.model.mass_shatter_model import COMPOSITION_FEATURE_DIM
from fragnnet.utils.frag_utils import CANONICAL_ELEMENT_ORDER
from tests.test_mass_shatter_model import _build_batch, _build_model


@pytest.fixture
def atom_embeddings() -> th.Tensor:
    th.manual_seed(11)
    return th.randn(4, 8)


def _endpoint_average_pool(embeddings: th.Tensor, pairs: list[tuple[int, int]]) -> th.Tensor:
    """Reproduce the default boundary embedding: mean of endpoint averages."""
    bond_embeddings = th.stack([(embeddings[u] + embeddings[v]) / 2.0 for u, v in pairs])
    return pool_boundary_bond_embeddings(
        bond_embeddings,
        th.arange(len(pairs)),
        th.zeros(len(pairs), dtype=th.long),
        num_components=1,
    )


def test_endpoint_average_boundary_pooling_is_degenerate(atom_embeddings: th.Tensor) -> None:
    """The default pooling loses both the pairing and the retained/lost distinction."""
    # Atoms 0, 1 inside the fragment; atoms 2, 3 outside it.
    paired = _endpoint_average_pool(atom_embeddings, [(0, 2), (1, 3)])
    crossed = _endpoint_average_pool(atom_embeddings, [(0, 3), (1, 2)])
    reversed_roles = _endpoint_average_pool(atom_embeddings, [(2, 0), (3, 1)])

    assert th.allclose(paired, crossed)
    assert th.allclose(paired, reversed_roles)


def test_cut_bond_encoder_preserves_pairing_and_orientation(atom_embeddings: th.Tensor) -> None:
    """The encoder distinguishes exactly what the average collapses."""
    th.manual_seed(5)
    encoder = CutBondEncoder(atom_dim=8, hidden_dim=12, out_dim=8)

    def pooled(pairs: list[tuple[int, int]]) -> th.Tensor:
        inside = th.tensor([u for u, _ in pairs])
        outside = th.tensor([v for _, v in pairs])
        return encoder(atom_embeddings[inside], atom_embeddings[outside]).mean(dim=0)

    paired = pooled([(0, 2), (1, 3)])
    crossed = pooled([(0, 3), (1, 2)])
    reversed_roles = pooled([(2, 0), (3, 1)])

    assert not th.allclose(paired, crossed)
    assert not th.allclose(paired, reversed_roles)


def test_boundary_orientation_splits_retained_and_lost_atoms() -> None:
    """Each boundary pair resolves to the atom inside its own component."""
    # Bond 0 is written (inside, outside) and bond 1 the other way round.
    bond_endpoints = th.tensor([[0, 3], [2, 1]])
    inside, outside = boundary_bond_endpoint_orientation(
        bond_endpoints,
        th.tensor([0, 1]),
        th.tensor([0, 0]),
        th.tensor([0, 1]),
        th.tensor([0, 0]),
        num_atoms=4,
    )

    assert inside.tolist() == [0, 1]
    assert outside.tolist() == [2, 3]


def test_boundary_orientation_rejects_a_bond_inside_its_component() -> None:
    """A bond with both endpoints retained is not a boundary bond."""
    with pytest.raises(ValueError, match="exactly one endpoint inside"):
        boundary_bond_endpoint_orientation(
            th.tensor([[0], [1]]),
            th.tensor([0]),
            th.tensor([0]),
            th.tensor([0, 1]),
            th.tensor([0, 0]),
            num_atoms=4,
        )


def test_model_boundary_embeddings_use_the_oriented_encoder() -> None:
    """``_boundary_embeddings`` routes through CutBondEncoder with the right orientation."""
    batch = _build_batch(("CCO",))
    model = _build_model(batch, boundary_bond_encoder=True)
    th.manual_seed(3)
    embeddings = th.randn(batch.x.shape[0], 16)
    endpoint_average = (
        embeddings[batch.chemical_bond_index[0]] + embeddings[batch.chemical_bond_index[1]]
    ) / 2.0
    # Fragment 0 keeps atom 0; its single boundary bond is bond 0 = (0, 1).
    pooled = model._boundary_embeddings(
        batch,
        embeddings,
        endpoint_average,
        th.tensor([0]),
        th.tensor([0]),
        th.tensor([0]),
        th.tensor([0]),
        1,
    )
    expected = model.boundary_bond_encoder(embeddings[[0]], embeddings[[1]])

    assert th.allclose(pooled, expected)
    assert not th.allclose(pooled, endpoint_average[[0]])


def test_candidate_mass_features_are_log_mass_and_precursor_fraction() -> None:
    """The whole molecule scores its own exact mass and a fraction of one."""
    smiles = "CCO"
    batch = _build_batch((smiles,))
    model = _build_model(batch, candidate_mass_features=True)
    num_atoms = Chem.MolFromSmiles(smiles).GetNumAtoms()
    mass_features, charge_feature, _, _ = model._candidate_scalar_features(
        batch,
        th.arange(num_atoms),
        th.zeros(num_atoms, dtype=th.long),
        th.tensor([0]),
        1,
        1,
    )

    assert charge_feature is None
    assert mass_features.shape == (1, 2)
    expected_mass = ExactMolWt(Chem.MolFromSmiles(smiles))
    assert mass_features[0, 0].item() == pytest.approx(th.log1p(th.tensor(expected_mass)).item())
    assert mass_features[0, 1].item() == pytest.approx(1.0)


def test_candidate_mass_fraction_tracks_fragment_size() -> None:
    """A proper fragment scores strictly below the intact precursor."""
    batch = _build_batch(("CCO",))
    model = _build_model(batch, candidate_mass_features=True)
    mass_features, _, _, _ = model._candidate_scalar_features(
        batch,
        th.tensor([0, 1, 2, 0]),
        th.tensor([0, 0, 0, 1]),
        th.tensor([0, 0]),
        2,
        1,
    )

    assert mass_features[1, 1].item() < mass_features[0, 1].item() == pytest.approx(1.0)
    assert 0.0 < mass_features[1, 1].item() < 1.0


def test_candidate_charge_feature_sums_gasteiger_charges() -> None:
    """Heavy-atom charges only, matching FraGNNet's ``frag_q``.

    RDKit keeps an atom's implicit-hydrogen charge in ``_GasteigerHCharge``, which neither model
    reads, so a neutral molecule's heavy-atom sum is not zero.
    """
    batch = _build_batch(("CCO",))
    model = _build_model(batch, candidate_charge_feature=True)
    _, charge_feature, _, _ = model._candidate_scalar_features(
        batch,
        th.tensor([0, 1, 2, 2]),
        th.tensor([0, 0, 0, 1]),
        th.tensor([0, 0]),
        2,
        1,
    )

    assert charge_feature.shape == (2,)
    assert charge_feature[0].item() == pytest.approx(
        batch.atom_gasteiger_charges.sum().item(), abs=1e-6
    )
    assert charge_feature[1].item() == pytest.approx(
        batch.atom_gasteiger_charges[2].item(), abs=1e-6
    )
    # The oxygen is the electronegative site, and a proper fragment is not the whole molecule.
    assert charge_feature[1].item() < 0.0
    assert charge_feature[1].item() != pytest.approx(charge_feature[0].item(), abs=1e-4)


def test_complement_composition_is_the_precursor_minus_the_candidate() -> None:
    """Ethanol C2H6O: the whole molecule loses nothing, a C fragment loses CH6O."""
    batch = _build_batch(("CCO",))
    model = _build_model(batch, candidate_composition_features=True)
    carbon_column = CANONICAL_ELEMENT_ORDER.index("C")
    hydrogen_column = CANONICAL_ELEMENT_ORDER.index("H")
    oxygen_column = CANONICAL_ELEMENT_ORDER.index("O")

    # Component 0 is the intact molecule; component 1 is the single terminal carbon.
    *_, composition = model._candidate_scalar_features(
        batch,
        th.tensor([0, 1, 2, 0]),
        th.tensor([0, 0, 0, 1]),
        th.tensor([0, 0]),
        2,
        1,
    )

    assert composition.shape == (2, COMPOSITION_FEATURE_DIM)
    counts = th.expm1(composition).round().long()
    assert counts[0].sum().item() == 0, "the intact precursor loses nothing"
    # Dropping the methyl carbon and its three hydrogens leaves CH3O behind.
    assert counts[1, carbon_column].item() == 1
    assert counts[1, oxygen_column].item() == 1
    assert counts[1, hydrogen_column].item() == 3
    assert counts[1].sum().item() == 5


def test_complement_composition_separates_isobaric_neutral_losses() -> None:
    """CO and C2H4 are 0.036 Da apart, inside one 0.1 Da bin, and differ only in formula.

    This is the case GLACIER's ``psi(r_form - f_k)`` exists for. Pentan-2-one has candidates
    for both: dropping the carbonyl {C1, O2} leaves CO behind, dropping the internal
    {C3, C4} leaves C2H4. The two candidates are themselves near-isobaric (C4H10 58.078 vs
    C3H6O 58.042), so the mass scalars barely move and cannot tell the losses apart.
    """
    batch = _build_batch(("CC(=O)CCC",))
    model = _build_model(batch, candidate_mass_features=True, candidate_composition_features=True)
    carbon_column = CANONICAL_ELEMENT_ORDER.index("C")
    hydrogen_column = CANONICAL_ELEMENT_ORDER.index("H")
    oxygen_column = CANONICAL_ELEMENT_ORDER.index("O")

    # Candidate 0 keeps {0, 3, 4, 5} and loses CO; candidate 1 keeps {0, 1, 2, 5} and loses C2H4.
    mass_features, _, _, composition = model._candidate_scalar_features(
        batch,
        th.tensor([0, 3, 4, 5, 0, 1, 2, 5]),
        th.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        th.tensor([0, 0]),
        2,
        1,
    )
    counts = th.expm1(composition).round().long()

    # CO: one carbon, one oxygen, no hydrogen.
    assert counts[0, carbon_column].item() == 1
    assert counts[0, oxygen_column].item() == 1
    assert counts[0, hydrogen_column].item() == 0
    # C2H4: two carbons, four hydrogens, no oxygen.
    assert counts[1, carbon_column].item() == 2
    assert counts[1, oxygen_column].item() == 0
    assert counts[1, hydrogen_column].item() == 4

    # The scalars the head had before this feature put the two candidates in the same 0.1 Da
    # bin; the composition rows are different vectors.
    candidate_masses = th.expm1(mass_features[:, 0])
    assert abs(candidate_masses[0].item() - candidate_masses[1].item()) < 0.1
    assert not th.equal(counts[0], counts[1])


def test_complement_composition_is_off_by_default() -> None:
    """A new input must not ride into a running comparison."""
    batch = _build_batch(("CCO",))
    model = _build_model(batch)
    *_, composition = model._candidate_scalar_features(
        batch,
        th.arange(3),
        th.zeros(3, dtype=th.long),
        th.tensor([0]),
        1,
        1,
    )

    assert composition is None
    assert model.intensity_head.composition_feature_dim == 0


def test_complement_composition_counts_every_precursor_separately() -> None:
    """Each candidate's complement comes from its own molecule, not the batch."""
    batch = _build_batch(("CCO", "c1ccccc1"))
    model = _build_model(batch, candidate_composition_features=True)
    carbon_column = CANONICAL_ELEMENT_ORDER.index("C")
    oxygen_column = CANONICAL_ELEMENT_ORDER.index("O")
    num_ethanol_atoms = Chem.MolFromSmiles("CCO").GetNumAtoms()

    # One candidate per molecule, each holding a single carbon.
    *_, composition = model._candidate_scalar_features(
        batch,
        th.tensor([0, num_ethanol_atoms]),
        th.tensor([0, 1]),
        th.tensor([0, 1]),
        2,
        2,
    )
    counts = th.expm1(composition).round().long()

    assert counts[0, oxygen_column].item() == 1, "ethanol's complement keeps the oxygen"
    assert counts[1, oxygen_column].item() == 0, "benzene has no oxygen to lose"
    assert counts[1, carbon_column].item() == 5, "benzene minus one carbon"


@pytest.mark.parametrize("unique_fragment_tokens", [False, True])
@pytest.mark.parametrize("use_base_probability_feature", [False, True])
@pytest.mark.parametrize(
    ("use_mass_features", "use_charge_feature"), [(False, False), (True, True)]
)
@pytest.mark.parametrize("composition_feature_dim", [0, COMPOSITION_FEATURE_DIM])
def test_intensity_head_input_width_tracks_enabled_features(
    unique_fragment_tokens: bool,
    use_base_probability_feature: bool,
    use_mass_features: bool,
    use_charge_feature: bool,
    composition_feature_dim: int,
) -> None:
    """Constant inputs are dropped in unique mode and the new scalars are added."""
    head = CandidateIntensityTransformerHead(
        fragment_embedding_dim=8,
        context_dim=4,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=8,
        num_inducing_points=2,
        num_ion_modes=1,
        num_h_shifts=3,
        dropout=0.0,
        use_base_probability_feature=use_base_probability_feature,
        use_mass_features=use_mass_features,
        use_charge_feature=use_charge_feature,
        composition_feature_dim=composition_feature_dim,
        unique_fragment_tokens=unique_fragment_tokens,
    )
    expected = 3 * 8 + 4 + 2  # embeddings, context, cut count, log atom count
    if not unique_fragment_tokens:
        expected += 1  # log per-mask component count, constant when each token is its own mask
    if use_base_probability_feature:
        expected += 1 if unique_fragment_tokens else 2
    if use_mass_features:
        expected += 2
    if use_charge_feature:
        expected += 1
    expected += composition_feature_dim

    assert head.input_projection[0].in_features == expected


def test_intensity_head_requires_the_scalars_it_declares() -> None:
    """Declaring a scalar and not supplying it is an error, not a silent zero."""
    head = CandidateIntensityTransformerHead(
        fragment_embedding_dim=4,
        context_dim=0,
        model_dim=4,
        num_layers=1,
        num_heads=2,
        feedforward_dim=4,
        num_inducing_points=2,
        num_ion_modes=1,
        num_h_shifts=1,
        dropout=0.0,
        use_base_probability_feature=False,
        use_mass_features=True,
    )
    embeddings = th.zeros(2, 4)
    with pytest.raises(ValueError, match="mass_features must be supplied"):
        head(
            embeddings,
            embeddings,
            embeddings,
            th.ones(2),
            th.ones(2, dtype=th.long),
            th.zeros(2),
            th.zeros(2),
            th.zeros(2, dtype=th.long),
            th.arange(2),
            batch_size=1,
            num_masks=2,
        )


@pytest.mark.parametrize("unique_fragment_mode", [False, True])
def test_forward_runs_with_every_candidate_feature_enabled(unique_fragment_mode: bool) -> None:
    """End-to-end forward in both candidate modes with the new defaults switched on."""
    batch = _build_batch(("CCOC", "CCO"))
    model = _build_model(
        batch,
        max_masks=4,
        max_enumerated_masks=8,
        unique_fragment_mode=unique_fragment_mode,
        boundary_bond_encoder=True,
        candidate_mass_features=True,
        candidate_charge_feature=True,
        candidate_composition_features=True,
    )
    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0, 0]),
        spec_nce=th.tensor([10.0, 40.0]),
    )

    assert th.isfinite(output["pred_logprobs"]).all()
    output["pred_logprobs"].mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.boundary_bond_encoder.parameters()
    )


def test_forward_no_longer_requires_atom_formal_charges() -> None:
    """The field was required by the forward check and read by nothing."""
    batch = _build_batch(("CCO",))
    model = _build_model(batch)
    del batch.atom_formal_charges

    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0]),
        spec_nce=th.tensor([20.0]),
    )

    assert th.isfinite(output["pred_logprobs"]).all()
