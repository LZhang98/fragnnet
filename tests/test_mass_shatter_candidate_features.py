"""Boundary-bond encoding and the mass/charge candidate scalars.

Covers the fixes recorded in ``docs/reference/masshatter_feature_set_audit.md`` §7.2, §7.4,
§7.5 and §7.7: the degenerate endpoint-average boundary embedding, the mass-blind candidate
token, the constant token columns in unique-fragment mode, and the unread required graph field.

Also covers the two 2026-09-22 candidate-token changes from
``docs/plans/masshatter_cut_set_features_plan.md``: the severed bonds' own input features,
which mean pooling over atom endpoints destroys, and per-atom dropout on the component pool,
which attacks the identity shortcut that pooling leaves behind.
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
    pool_component_and_complement_embeddings,
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
        candidate_cut_bond_features=True,
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


def test_cut_bond_features_sum_and_average_the_severed_bonds() -> None:
    """Phenethyl alcohol: two single cuts and two aromatic cuts, same count, different bonds."""
    batch = _build_batch(("c1ccccc1CCO",))
    model = _build_model(batch, candidate_cut_bond_features=True)
    single = batch.chemical_bond_attr[5]
    aromatic = batch.chemical_bond_attr[0]
    assert not th.allclose(single, aromatic), "the fixture must separate the two bond orders"

    # Candidate 0 is cut at the two single C-C bonds, candidate 1 at two bonds of one ring.
    cut_features = model._cut_bond_features(
        batch,
        th.tensor([5, 6, 0, 1]),
        th.tensor([0, 0, 1, 1]),
        2,
    )

    width = batch.chemical_bond_attr.shape[1]
    assert cut_features.shape == (2, 2 * width)
    assert th.allclose(cut_features[0, :width], 2.0 * single)
    assert th.allclose(cut_features[0, width:], single)
    assert th.allclose(cut_features[1, :width], 2.0 * aromatic)
    assert th.allclose(cut_features[1, width:], aromatic)
    # The head's existing cut description -- the count -- is 2 for both.
    assert not th.allclose(cut_features[0], cut_features[1])


def test_cut_bond_features_are_zero_without_a_severed_bond() -> None:
    """The intact precursor broke nothing, and scores nothing, rather than dividing by zero."""
    batch = _build_batch(("c1ccccc1CCO",))
    model = _build_model(batch, candidate_cut_bond_features=True)

    cut_features = model._cut_bond_features(
        batch,
        th.tensor([], dtype=th.long),
        th.tensor([], dtype=th.long),
        1,
    )

    assert cut_features.shape == (1, 2 * batch.chemical_bond_attr.shape[1])
    assert th.count_nonzero(cut_features) == 0


def test_cut_bond_features_are_off_by_default() -> None:
    """Off unless asked for, so the token width cannot move under a running comparison."""
    batch = _build_batch(("CCO",))
    model = _build_model(batch)

    assert model.intensity_head.cut_bond_feature_dim == 0
    assert model._cut_bond_features(batch, th.tensor([0]), th.tensor([0]), 1) is None


def test_cut_bond_features_widen_the_intensity_token() -> None:
    """Enabling the feature adds twice the bond width to the head's input."""
    batch = _build_batch(("CCO",))
    baseline = _build_model(batch)
    widened = _build_model(batch, candidate_cut_bond_features=True)
    bond_width = batch.edge_attr.shape[1]

    assert widened.intensity_head.cut_bond_feature_dim == 2 * bond_width
    assert (
        widened.intensity_head.input_projection[0].in_features
        - baseline.intensity_head.input_projection[0].in_features
        == 2 * bond_width
    )


@pytest.mark.parametrize("cut_bond_feature_dim", [0, 10])
def test_intensity_head_input_width_tracks_cut_bond_features(cut_bond_feature_dim: int) -> None:
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
        use_base_probability_feature=False,
        cut_bond_feature_dim=cut_bond_feature_dim,
    )
    # Embeddings, context, cut count, log atom count, log per-mask component count.
    expected = 3 * 8 + 4 + 2 + 1 + cut_bond_feature_dim

    assert head.input_projection[0].in_features == expected


def test_intensity_head_requires_declared_cut_bond_features() -> None:
    """Declaring the width and not supplying the tensor is an error, not a silent zero."""
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
        cut_bond_feature_dim=6,
    )
    embeddings = th.zeros(2, 4)
    with pytest.raises(ValueError, match="cut_bond_features must be supplied"):
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
    with pytest.raises(ValueError, match="cut_bond_features must have shape"):
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
            cut_bond_features=th.zeros(2, 5),
        )


def test_candidate_atom_dropout_is_inactive_outside_training() -> None:
    """Evaluation pools the true atom set, so validation metrics stay deterministic."""
    batch = _build_batch(("c1ccccc1CCO",))
    model = _build_model(batch, candidate_atom_dropout=0.5)
    model.eval()
    atom_indices = th.arange(9)
    component_indices = th.zeros(9, dtype=th.long)

    kept_atoms, kept_components = model._candidate_atom_dropout(atom_indices, component_indices, 1)

    assert th.equal(kept_atoms, atom_indices)
    assert th.equal(kept_components, component_indices)


def test_candidate_atom_dropout_is_inactive_at_probability_zero() -> None:
    """The default is exactly the old behavior, in training as well."""
    batch = _build_batch(("c1ccccc1CCO",))
    model = _build_model(batch)
    model.train()
    atom_indices = th.arange(9)
    component_indices = th.zeros(9, dtype=th.long)

    kept_atoms, kept_components = model._candidate_atom_dropout(atom_indices, component_indices, 1)

    assert th.equal(kept_atoms, atom_indices)
    assert th.equal(kept_components, component_indices)


def test_candidate_atom_dropout_drops_atoms_in_training() -> None:
    """Roughly the requested share of membership entries is removed."""
    batch = _build_batch(("c1ccccc1CCO",))
    model = _build_model(batch, candidate_atom_dropout=0.5)
    model.train()
    th.manual_seed(17)
    num_components = 200
    atom_indices = th.arange(9).repeat(num_components)
    component_indices = th.arange(num_components).repeat_interleave(9)

    kept_atoms, kept_components = model._candidate_atom_dropout(
        atom_indices, component_indices, num_components
    )

    assert kept_atoms.shape == kept_components.shape
    kept_fraction = kept_atoms.numel() / atom_indices.numel()
    assert 0.45 < kept_fraction < 0.65, kept_fraction
    # Restoring emptied components can only raise the kept share above the nominal 0.5.
    assert kept_fraction > 0.5 - 0.05
    sizes = th.bincount(kept_components, minlength=num_components)
    assert (sizes > 0).all(), "an empty pool has no mean"


def test_candidate_atom_dropout_restores_a_candidate_it_emptied() -> None:
    """At an extreme rate every candidate still pools its true atom set rather than none."""
    batch = _build_batch(("CCO",))
    model = _build_model(batch, candidate_atom_dropout=0.99)
    model.train()
    th.manual_seed(3)
    atom_indices = th.tensor([0, 1, 2, 0, 1, 2])
    component_indices = th.tensor([0, 0, 0, 1, 1, 1])

    kept_atoms, kept_components = model._candidate_atom_dropout(atom_indices, component_indices, 2)

    sizes = th.bincount(kept_components, minlength=2)
    assert (sizes > 0).all()
    for component in range(2):
        kept = kept_atoms[kept_components == component]
        # A component that lost every atom is restored whole, never left with a random rump.
        if kept.numel() < 3:
            assert kept.numel() >= 1
        else:
            assert sorted(kept.tolist()) == [0, 1, 2]


def test_candidate_atom_dropout_keeps_the_precursor_partition() -> None:
    """Dropped atoms move to the complement, so neither pool leaks the true atom set.

    The complement is the precursor minus the candidate, so an exact complement beside a
    corrupted candidate would leave the true candidate sum recoverable by subtraction.
    """
    batch = _build_batch(("c1ccccc1CCO",))
    model = _build_model(batch, candidate_atom_dropout=0.5)
    model.train()
    th.manual_seed(29)
    atom_embeddings = th.randn(batch.x.shape[0], 6)
    atom_indices = th.arange(9)
    component_indices = th.zeros(9, dtype=th.long)
    kept_atoms, kept_components = model._candidate_atom_dropout(atom_indices, component_indices, 1)
    assert kept_atoms.numel() < atom_indices.numel(), "the fixture must actually drop something"

    component_sums, complement_sums = pool_component_and_complement_embeddings(
        atom_embeddings,
        kept_atoms,
        kept_components,
        th.zeros(1, dtype=th.long),
        batch.batch,
        num_components=1,
        num_molecules=1,
        reduction="sum",
    )
    exact_sums, _ = pool_component_and_complement_embeddings(
        atom_embeddings,
        atom_indices,
        component_indices,
        th.zeros(1, dtype=th.long),
        batch.batch,
        num_components=1,
        num_molecules=1,
        reduction="sum",
    )

    assert th.allclose(component_sums + complement_sums, atom_embeddings.sum(dim=0), atol=1e-5)
    assert not th.allclose(component_sums, exact_sums)


def test_candidate_atom_dropout_rejects_certain_dropout() -> None:
    batch = _build_batch(("CCO",))
    with pytest.raises(ValueError, match="candidate_atom_dropout must be in"):
        _build_model(batch, candidate_atom_dropout=1.0)


@pytest.mark.parametrize("unique_fragment_mode", [False, True])
def test_forward_runs_with_cut_features_and_atom_dropout(unique_fragment_mode: bool) -> None:
    """Both candidate modes train with the new token column and the pooled-atom dropout."""
    batch = _build_batch(("CCOC", "CCO"))
    model = _build_model(
        batch,
        max_masks=4,
        max_enumerated_masks=8,
        unique_fragment_mode=unique_fragment_mode,
        candidate_cut_bond_features=True,
        candidate_atom_dropout=0.3,
    )
    model.train()
    th.manual_seed(7)

    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0, 0]),
        spec_nce=th.tensor([10.0, 40.0]),
    )

    assert th.isfinite(output["pred_logprobs"]).all()
    output["pred_logprobs"].mean().backward()
    assert model.intensity_head.input_projection[0].weight.grad.abs().sum() > 0


def test_cut_adjacency_separates_a_retained_from_a_lost_shared_atom() -> None:
    """Propane: keeping the middle carbon and losing it are the same two cut bonds.

    Component 0 keeps atom 1, so both cuts meet at a RETAINED atom -- one atom lost two
    substituents. Component 1 keeps atoms 0 and 2, so the same two cuts meet at a LOST atom --
    the fragment expelled a bridging CH2. Every pooled cut descriptor represents these
    identically; this feature does not.
    """
    batch = _build_batch(("CCC",))
    model = _build_model(batch, candidate_cut_adjacency_features=True)

    adjacency = model._cut_adjacency_features(
        batch,
        th.tensor([0, 1, 0, 1]),
        th.tensor([0, 0, 1, 1]),
        th.tensor([1, 0, 2]),
        th.tensor([0, 1, 1]),
        2,
    )

    assert adjacency.shape == (2, 2)
    assert adjacency[0].tolist() == [1.0, 0.0]
    assert adjacency[1].tolist() == [0.0, 1.0]


def test_cut_adjacency_is_zero_for_a_ring_opening() -> None:
    """Benzene: two cuts in one ring are not adjacent, so ring sharing is a different event."""
    batch = _build_batch(("c1ccccc1",))
    model = _build_model(batch, candidate_cut_adjacency_features=True)

    # Component 0 keeps atoms 0-2: cuts are bonds (2,3) and (5,0), which share no atom.
    # Component 1 keeps atom 0 alone: cuts are bonds (0,1) and (5,0), meeting at atom 0.
    adjacency = model._cut_adjacency_features(
        batch,
        th.tensor([2, 5, 0, 5]),
        th.tensor([0, 0, 1, 1]),
        th.tensor([0, 1, 2, 0]),
        th.tensor([0, 0, 0, 1]),
        2,
    )

    assert adjacency[0].tolist() == [0.0, 0.0]
    assert adjacency[1].tolist() == [1.0, 0.0]


def test_cut_adjacency_is_zero_for_a_single_cut() -> None:
    """One severed bond has no pair to be adjacent to, and no division by zero."""
    batch = _build_batch(("CCO",))
    model = _build_model(batch, candidate_cut_adjacency_features=True)

    adjacency = model._cut_adjacency_features(
        batch,
        th.tensor([0]),
        th.tensor([0]),
        th.tensor([0]),
        th.tensor([0]),
        1,
    )

    assert adjacency.shape == (1, 2)
    assert th.count_nonzero(adjacency) == 0


def test_cut_adjacency_is_off_by_default_and_widens_the_token_by_two() -> None:
    batch = _build_batch(("CCO",))
    baseline = _build_model(batch)
    widened = _build_model(batch, candidate_cut_adjacency_features=True)

    assert baseline.intensity_head.cut_adjacency_feature_dim == 0
    assert (
        baseline._cut_adjacency_features(
            batch, th.tensor([0]), th.tensor([0]), th.tensor([0]), th.tensor([0]), 1
        )
        is None
    )
    assert widened.intensity_head.cut_adjacency_feature_dim == 2
    assert (
        widened.intensity_head.input_projection[0].in_features
        - baseline.intensity_head.input_projection[0].in_features
        == 2
    )


def test_intensity_head_requires_declared_cut_adjacency_features() -> None:
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
        cut_adjacency_feature_dim=2,
    )
    embeddings = th.zeros(2, 4)
    with pytest.raises(ValueError, match="cut_adjacency_features must be supplied"):
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
def test_forward_runs_with_the_full_cut_description(unique_fragment_mode: bool) -> None:
    """Both cut blocks and the pooled-atom dropout together, in both candidate modes."""
    batch = _build_batch(("CCOC", "CCO"))
    model = _build_model(
        batch,
        max_masks=4,
        max_enumerated_masks=8,
        unique_fragment_mode=unique_fragment_mode,
        boundary_bond_encoder=True,
        candidate_cut_bond_features=True,
        candidate_cut_adjacency_features=True,
        candidate_atom_dropout=0.2,
    )
    model.train()
    th.manual_seed(11)

    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0, 0]),
        spec_nce=th.tensor([10.0, 40.0]),
    )

    assert th.isfinite(output["pred_logprobs"]).all()
    output["pred_logprobs"].mean().backward()
    assert model.intensity_head.input_projection[0].weight.grad.abs().sum() > 0
