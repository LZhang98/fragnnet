"""Boundary-bond encoding and the candidate-feature descriptor columns.

Covers the fixes recorded in ``docs/reference/masster_feature_set_audit.md`` §7.2, §7.4,
§7.5 and §7.7: the degenerate endpoint-average boundary embedding, the mass-blind candidate
token, the constant token columns in unique-fragment mode, and the unread required graph field.

Also covers the two 2026-09-22 candidate-token changes from
``docs/plans/masster/masster_cut_set_features_plan.md``: the severed bonds' own input features,
which mean pooling over atom endpoints destroys, and per-atom dropout on the component pool,
which attacks the identity shortcut that pooling leaves behind.

And the 2026-09-22 refactor that replaced the five independent
``candidate_mass_features``/``candidate_charge_feature``/``candidate_composition_features``/
``candidate_cut_bond_features``/``candidate_cut_adjacency_features`` booleans with the single
``candidate_features`` list resolved against
:mod:`fragnnet.models.masster.candidate_features`. Most of this file builds a
:class:`~fragnnet.models.masster.candidate_features.CandidateFeatureInputs` directly and
calls a spec's ``build`` (or
:func:`~fragnnet.models.masster.candidate_features.build_candidate_features`) instead of the
deleted private model methods; the asserted values are unchanged from before the refactor.
"""

import pytest
import torch as th
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from torch_geometric.data import Batch

import fragnnet.models.masster.masster_model as masster_model_module
from fragnnet.layers.cut_bond_encoder import CutBondEncoder
from fragnnet.models.masster.bond_fragment_heads import (
    CandidateIntensityTransformerHead,
    boundary_bond_endpoint_orientation,
    pool_boundary_bond_embeddings,
    pool_component_and_complement_embeddings,
)
from fragnnet.models.masster.candidate_features import (
    CANDIDATE_FEATURES_BY_NAME,
    COMPOSITION_FEATURE_DIM,
    CandidateFeatureInputs,
    build_candidate_features,
    candidate_feature_required_graph_fields,
    candidate_feature_width,
    describe_candidate_features,
    resolve_candidate_features,
)
from fragnnet.models.masster.masster_model import MassterModel
from fragnnet.utils.formula_utils import CANONICAL_ELEMENT_ORDER
from tests.test_masster_model import _build_batch, _build_model


def _candidate_inputs(
    model: MassterModel,
    batch: Batch,
    *,
    atom_indices: th.Tensor,
    atom_component_indices: th.Tensor,
    component_batch_indices: th.Tensor,
    num_components: int,
    batch_size: int,
    boundary_bond_indices: th.Tensor | None = None,
    boundary_component_indices: th.Tensor | None = None,
    cut_counts: th.Tensor | None = None,
    dtype: th.dtype = th.float32,
) -> CandidateFeatureInputs:
    """Build a :class:`CandidateFeatureInputs` for one spec's ``build``, filling in the fields a
    given spec does not read with harmless placeholders.
    """
    return CandidateFeatureInputs(
        mol_pyg=batch,
        atom_indices=atom_indices,
        atom_component_indices=atom_component_indices,
        boundary_bond_indices=(
            th.tensor([], dtype=th.long)
            if boundary_bond_indices is None
            else boundary_bond_indices
        ),
        boundary_component_indices=(
            th.tensor([], dtype=th.long)
            if boundary_component_indices is None
            else boundary_component_indices
        ),
        component_batch_indices=component_batch_indices,
        cut_counts=(
            th.zeros(num_components, dtype=th.long) if cut_counts is None else cut_counts
        ),
        composition_element_columns=model.composition_element_columns,
        num_components=num_components,
        batch_size=batch_size,
        dtype=dtype,
    )


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


def test_forward_boundary_dedup_matches_per_row_encoding(monkeypatch) -> None:
    """Encoding each directed cut bond once reproduces encoding every (fragment, bond) row."""
    batch = _build_batch(("CCOC(=O)c1ccccc1", "CCO"))
    model = _build_model(
        batch,
        max_masks=16,
        max_enumerated_masks=64,
        unique_fragment_mode=True,
        boundary_bond_encoder=True,
    ).eval()
    inputs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 0]),
        "spec_nce": th.tensor([20.0, 40.0]),
    }
    encoded_rows = []
    handle = model.boundary_bond_encoder.register_forward_hook(
        lambda _module, args, _output: encoded_rows.append(len(args[0]))
    )
    with th.no_grad():
        deduplicated = model(**inputs)["pred_logprobs"]
        monkeypatch.setattr(
            masster_model_module,
            "encode_cut_bond_pairs",
            lambda encoder, atoms, in_idx, out_idx: encoder(atoms[in_idx], atoms[out_idx]),
        )
        per_row = model(**inputs)["pred_logprobs"]
    handle.remove()

    assert encoded_rows[0] < encoded_rows[1]  # the batch repeats directed bonds: not vacuous
    assert th.allclose(deduplicated, per_row, atol=1e-6)


# --- c_mass -------------------------------------------------------------------------------


def test_candidate_mass_features_are_log_mass_and_precursor_fraction() -> None:
    """The whole molecule scores its own exact mass and a fraction of one."""
    smiles = "CCO"
    batch = _build_batch((smiles,))
    model = _build_model(batch, candidate_features=("c_cut_count", "c_size", "c_mass"))
    num_atoms = Chem.MolFromSmiles(smiles).GetNumAtoms()
    inputs = _candidate_inputs(
        model,
        batch,
        atom_indices=th.arange(num_atoms),
        atom_component_indices=th.zeros(num_atoms, dtype=th.long),
        component_batch_indices=th.tensor([0]),
        num_components=1,
        batch_size=1,
    )
    mass_features = CANDIDATE_FEATURES_BY_NAME["c_mass"].build(inputs)

    assert mass_features.shape == (1, 2)
    expected_mass = ExactMolWt(Chem.MolFromSmiles(smiles))
    assert mass_features[0, 0].item() == pytest.approx(th.log1p(th.tensor(expected_mass)).item())
    assert mass_features[0, 1].item() == pytest.approx(1.0)


def test_candidate_mass_fraction_tracks_fragment_size() -> None:
    """A proper fragment scores strictly below the intact precursor."""
    batch = _build_batch(("CCO",))
    model = _build_model(batch, candidate_features=("c_cut_count", "c_size", "c_mass"))
    inputs = _candidate_inputs(
        model,
        batch,
        atom_indices=th.tensor([0, 1, 2, 0]),
        atom_component_indices=th.tensor([0, 0, 0, 1]),
        component_batch_indices=th.tensor([0, 0]),
        num_components=2,
        batch_size=1,
    )
    mass_features = CANDIDATE_FEATURES_BY_NAME["c_mass"].build(inputs)

    assert mass_features[1, 1].item() < mass_features[0, 1].item() == pytest.approx(1.0)
    assert 0.0 < mass_features[1, 1].item() < 1.0


# --- c_charge -------------------------------------------------------------------------------


def test_candidate_charge_feature_sums_gasteiger_charges() -> None:
    """Heavy-atom charges only, matching FraGNNet's ``frag_q``.

    RDKit keeps an atom's implicit-hydrogen charge in ``_GasteigerHCharge``, which neither model
    reads, so a neutral molecule's heavy-atom sum is not zero.
    """
    batch = _build_batch(("CCO",))
    model = _build_model(batch, candidate_features=("c_cut_count", "c_size", "c_charge"))
    inputs = _candidate_inputs(
        model,
        batch,
        atom_indices=th.tensor([0, 1, 2, 2]),
        atom_component_indices=th.tensor([0, 0, 0, 1]),
        component_batch_indices=th.tensor([0, 0]),
        num_components=2,
        batch_size=1,
    )
    charge_feature = CANDIDATE_FEATURES_BY_NAME["c_charge"].build(inputs).squeeze(1)

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


# --- c_nl_composition -----------------------------------------------------------------------


def test_complement_composition_is_the_precursor_minus_the_candidate() -> None:
    """Ethanol C2H6O: the whole molecule loses nothing, a C fragment loses CH6O."""
    batch = _build_batch(("CCO",))
    model = _build_model(batch, candidate_features=("c_cut_count", "c_size", "c_nl_composition"))
    carbon_column = CANONICAL_ELEMENT_ORDER.index("C")
    hydrogen_column = CANONICAL_ELEMENT_ORDER.index("H")
    oxygen_column = CANONICAL_ELEMENT_ORDER.index("O")

    # Component 0 is the intact molecule; component 1 is the single terminal carbon.
    inputs = _candidate_inputs(
        model,
        batch,
        atom_indices=th.tensor([0, 1, 2, 0]),
        atom_component_indices=th.tensor([0, 0, 0, 1]),
        component_batch_indices=th.tensor([0, 0]),
        num_components=2,
        batch_size=1,
    )
    composition = CANDIDATE_FEATURES_BY_NAME["c_nl_composition"].build(inputs)

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
    model = _build_model(
        batch,
        candidate_features=("c_cut_count", "c_size", "c_mass", "c_nl_composition"),
    )
    carbon_column = CANONICAL_ELEMENT_ORDER.index("C")
    hydrogen_column = CANONICAL_ELEMENT_ORDER.index("H")
    oxygen_column = CANONICAL_ELEMENT_ORDER.index("O")

    # Candidate 0 keeps {0, 3, 4, 5} and loses CO; candidate 1 keeps {0, 1, 2, 5} and loses C2H4.
    inputs = _candidate_inputs(
        model,
        batch,
        atom_indices=th.tensor([0, 3, 4, 5, 0, 1, 2, 5]),
        atom_component_indices=th.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        component_batch_indices=th.tensor([0, 0]),
        num_components=2,
        batch_size=1,
    )
    mass_features = CANDIDATE_FEATURES_BY_NAME["c_mass"].build(inputs)
    composition = CANDIDATE_FEATURES_BY_NAME["c_nl_composition"].build(inputs)
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


def test_complement_composition_counts_every_precursor_separately() -> None:
    """Each candidate's complement comes from its own molecule, not the batch."""
    batch = _build_batch(("CCO", "c1ccccc1"))
    model = _build_model(batch, candidate_features=("c_cut_count", "c_size", "c_nl_composition"))
    carbon_column = CANONICAL_ELEMENT_ORDER.index("C")
    oxygen_column = CANONICAL_ELEMENT_ORDER.index("O")
    num_ethanol_atoms = Chem.MolFromSmiles("CCO").GetNumAtoms()

    # One candidate per molecule, each holding a single carbon.
    inputs = _candidate_inputs(
        model,
        batch,
        atom_indices=th.tensor([0, num_ethanol_atoms]),
        atom_component_indices=th.tensor([0, 1]),
        component_batch_indices=th.tensor([0, 1]),
        num_components=2,
        batch_size=2,
    )
    composition = CANDIDATE_FEATURES_BY_NAME["c_nl_composition"].build(inputs)
    counts = th.expm1(composition).round().long()

    assert counts[0, oxygen_column].item() == 1, "ethanol's complement keeps the oxygen"
    assert counts[1, oxygen_column].item() == 0, "benzene has no oxygen to lose"
    assert counts[1, carbon_column].item() == 5, "benzene minus one carbon"


# --- default feature list / off-by-default coverage ------------------------------------------


def test_default_candidate_feature_list_is_cut_count_and_size_only() -> None:
    """A new input must not ride into a running comparison."""
    batch = _build_batch(("CCO",))
    model = _build_model(batch)

    assert [spec.name for spec in model.candidate_feature_specs] == ["c_cut_count", "c_size"]
    assert model.candidate_feature_dim == 2
    for extra in ("c_mass", "c_charge", "c_nl_composition", "c_cut_bonds", "c_cut_adjacency"):
        assert extra not in {spec.name for spec in model.candidate_feature_specs}


# --- head input width -------------------------------------------------------------------------


@pytest.mark.parametrize("unique_fragment_tokens", [False, True])
@pytest.mark.parametrize("use_base_probability_feature", [False, True])
@pytest.mark.parametrize("candidate_feature_dim", [0, 2, 5, COMPOSITION_FEATURE_DIM])
def test_intensity_head_input_width_tracks_candidate_feature_dim(
    unique_fragment_tokens: bool,
    use_base_probability_feature: bool,
    candidate_feature_dim: int,
) -> None:
    """Constant inputs are dropped in unique mode; the descriptor block is one summed width."""
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
        candidate_feature_dim=candidate_feature_dim,
        unique_fragment_tokens=unique_fragment_tokens,
    )
    expected = 3 * 8 + 4 + candidate_feature_dim  # embeddings, context, descriptor block
    if not unique_fragment_tokens:
        expected += 1  # log per-mask component count, constant when each token is its own mask
    if use_base_probability_feature:
        expected += 1 if unique_fragment_tokens else 2

    assert head.input_projection[0].in_features == expected


def test_production_candidate_feature_list_widens_head_to_1285() -> None:
    """The c4a_base_s1 geometry: mol_hidden_size 384, context 128, [count, size, mass, charge]."""
    specs = resolve_candidate_features(["c_cut_count", "c_size", "c_mass", "c_charge"])
    dim = candidate_feature_width(specs, edge_feature_dim=9)
    assert dim == 5

    head = CandidateIntensityTransformerHead(
        fragment_embedding_dim=384,
        context_dim=128,
        model_dim=16,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        num_inducing_points=2,
        num_ion_modes=1,
        num_h_shifts=3,
        dropout=0.0,
        use_base_probability_feature=False,
        candidate_feature_dim=dim,
        unique_fragment_tokens=True,
    )

    assert head.input_projection[0].in_features == 1285


def test_intensity_head_requires_declared_candidate_features() -> None:
    """Declaring the width and not supplying the tensor is an error, not a silent zero.

    Consolidates the three old per-feature "requires the scalars/features it declares" checks
    (mass/charge/composition, cut bonds, cut adjacency): there is now exactly one gate.
    """
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
        candidate_feature_dim=6,
    )
    embeddings = th.zeros(2, 4)
    with pytest.raises(ValueError, match="candidate_features must be supplied"):
        head(
            embeddings,
            embeddings,
            embeddings,
            th.zeros(2),
            th.zeros(2),
            th.zeros(2, dtype=th.long),
            th.arange(2),
            batch_size=1,
            num_masks=2,
        )
    with pytest.raises(ValueError, match="candidate_features must have shape"):
        head(
            embeddings,
            embeddings,
            embeddings,
            th.zeros(2),
            th.zeros(2),
            th.zeros(2, dtype=th.long),
            th.arange(2),
            batch_size=1,
            num_masks=2,
            candidate_features=th.zeros(2, 5),
        )


def test_empty_candidate_feature_list_head_runs_without_candidate_features() -> None:
    """``candidate_feature_dim=0`` is the new minimum -- the token carries no descriptor block."""
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
        candidate_feature_dim=0,
    )
    embeddings = th.randn(3, 8)

    outputs = head(
        embeddings,
        embeddings,
        embeddings,
        th.zeros(3),
        th.zeros(3),
        th.zeros(3, dtype=th.long),
        th.arange(3),
        batch_size=1,
        num_masks=3,
        context=th.zeros(3, 4),
    )

    assert th.isfinite(outputs[0]).all()


@pytest.mark.parametrize("unique_fragment_mode", [False, True])
def test_forward_runs_with_every_candidate_feature_enabled(unique_fragment_mode: bool) -> None:
    """End-to-end forward in both candidate modes with every descriptor but adjacency on."""
    batch = _build_batch(("CCOC", "CCO"))
    model = _build_model(
        batch,
        max_masks=4,
        max_enumerated_masks=8,
        unique_fragment_mode=unique_fragment_mode,
        boundary_bond_encoder=True,
        candidate_features=(
            "c_cut_count",
            "c_size",
            "c_mass",
            "c_charge",
            "c_nl_composition",
            "c_cut_bonds",
        ),
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


# --- c_cut_bonds ------------------------------------------------------------------------------


def test_cut_bond_features_sum_and_average_the_severed_bonds() -> None:
    """Phenethyl alcohol: two single cuts and two aromatic cuts, same count, different bonds."""
    batch = _build_batch(("c1ccccc1CCO",))
    model = _build_model(batch, candidate_features=("c_cut_count", "c_size", "c_cut_bonds"))
    single = batch.chemical_bond_attr[5]
    aromatic = batch.chemical_bond_attr[0]
    assert not th.allclose(single, aromatic), "the fixture must separate the two bond orders"

    # Candidate 0 is cut at the two single C-C bonds, candidate 1 at two bonds of one ring.
    inputs = _candidate_inputs(
        model,
        batch,
        atom_indices=th.tensor([], dtype=th.long),
        atom_component_indices=th.tensor([], dtype=th.long),
        component_batch_indices=th.tensor([0, 0]),
        num_components=2,
        batch_size=1,
        boundary_bond_indices=th.tensor([5, 6, 0, 1]),
        boundary_component_indices=th.tensor([0, 0, 1, 1]),
    )
    cut_features = CANDIDATE_FEATURES_BY_NAME["c_cut_bonds"].build(inputs)

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
    model = _build_model(batch, candidate_features=("c_cut_count", "c_size", "c_cut_bonds"))

    inputs = _candidate_inputs(
        model,
        batch,
        atom_indices=th.tensor([], dtype=th.long),
        atom_component_indices=th.tensor([], dtype=th.long),
        component_batch_indices=th.tensor([0]),
        num_components=1,
        batch_size=1,
        boundary_bond_indices=th.tensor([], dtype=th.long),
        boundary_component_indices=th.tensor([], dtype=th.long),
    )
    cut_features = CANDIDATE_FEATURES_BY_NAME["c_cut_bonds"].build(inputs)

    assert cut_features.shape == (1, 2 * batch.chemical_bond_attr.shape[1])
    assert th.count_nonzero(cut_features) == 0


def test_cut_bond_features_widen_the_intensity_token() -> None:
    """Enabling the feature adds twice the bond width to the head's input."""
    batch = _build_batch(("CCO",))
    baseline = _build_model(batch)
    widened = _build_model(batch, candidate_features=("c_cut_count", "c_size", "c_cut_bonds"))
    bond_width = batch.chemical_bond_attr.shape[1]

    assert widened.candidate_feature_dim - baseline.candidate_feature_dim == 2 * bond_width
    assert (
        widened.intensity_head.input_projection[0].in_features
        - baseline.intensity_head.input_projection[0].in_features
        == 2 * bond_width
    )


@pytest.mark.parametrize("unique_fragment_mode", [False, True])
def test_forward_runs_with_cut_features_and_atom_dropout(unique_fragment_mode: bool) -> None:
    """Both candidate modes train with the new token column and the pooled-atom dropout."""
    batch = _build_batch(("CCOC", "CCO"))
    model = _build_model(
        batch,
        max_masks=4,
        max_enumerated_masks=8,
        unique_fragment_mode=unique_fragment_mode,
        candidate_features=("c_cut_count", "c_size", "c_cut_bonds"),
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


# --- c_cut_adjacency --------------------------------------------------------------------------


def test_cut_adjacency_separates_a_retained_from_a_lost_shared_atom() -> None:
    """Propane: keeping the middle carbon and losing it are the same two cut bonds.

    Component 0 keeps atom 1, so both cuts meet at a RETAINED atom -- one atom lost two
    substituents. Component 1 keeps atoms 0 and 2, so the same two cuts meet at a LOST atom --
    the fragment expelled a bridging CH2. Every pooled cut descriptor represents these
    identically; this feature does not.
    """
    batch = _build_batch(("CCC",))
    model = _build_model(batch, candidate_features=("c_cut_count", "c_size", "c_cut_adjacency"))

    inputs = _candidate_inputs(
        model,
        batch,
        atom_indices=th.tensor([1, 0, 2]),
        atom_component_indices=th.tensor([0, 1, 1]),
        component_batch_indices=th.tensor([0, 0]),
        num_components=2,
        batch_size=1,
        boundary_bond_indices=th.tensor([0, 1, 0, 1]),
        boundary_component_indices=th.tensor([0, 0, 1, 1]),
    )
    adjacency = CANDIDATE_FEATURES_BY_NAME["c_cut_adjacency"].build(inputs)

    assert adjacency.shape == (2, 2)
    assert adjacency[0].tolist() == [1.0, 0.0]
    assert adjacency[1].tolist() == [0.0, 1.0]


def test_cut_adjacency_is_zero_for_a_ring_opening() -> None:
    """Benzene: two cuts in one ring are not adjacent, so ring sharing is a different event."""
    batch = _build_batch(("c1ccccc1",))
    model = _build_model(batch, candidate_features=("c_cut_count", "c_size", "c_cut_adjacency"))

    # Component 0 keeps atoms 0-2: cuts are bonds (2,3) and (5,0), which share no atom.
    # Component 1 keeps atom 0 alone: cuts are bonds (0,1) and (5,0), meeting at atom 0.
    inputs = _candidate_inputs(
        model,
        batch,
        atom_indices=th.tensor([0, 1, 2, 0]),
        atom_component_indices=th.tensor([0, 0, 0, 1]),
        component_batch_indices=th.tensor([0, 0]),
        num_components=2,
        batch_size=1,
        boundary_bond_indices=th.tensor([2, 5, 0, 5]),
        boundary_component_indices=th.tensor([0, 0, 1, 1]),
    )
    adjacency = CANDIDATE_FEATURES_BY_NAME["c_cut_adjacency"].build(inputs)

    assert adjacency[0].tolist() == [0.0, 0.0]
    assert adjacency[1].tolist() == [1.0, 0.0]


def test_cut_adjacency_is_zero_for_a_single_cut() -> None:
    """One severed bond has no pair to be adjacent to, and no division by zero."""
    batch = _build_batch(("CCO",))
    model = _build_model(batch, candidate_features=("c_cut_count", "c_size", "c_cut_adjacency"))

    inputs = _candidate_inputs(
        model,
        batch,
        atom_indices=th.tensor([0]),
        atom_component_indices=th.tensor([0]),
        component_batch_indices=th.tensor([0]),
        num_components=1,
        batch_size=1,
        boundary_bond_indices=th.tensor([0]),
        boundary_component_indices=th.tensor([0]),
    )
    adjacency = CANDIDATE_FEATURES_BY_NAME["c_cut_adjacency"].build(inputs)

    assert adjacency.shape == (1, 2)
    assert th.count_nonzero(adjacency) == 0


def test_cut_adjacency_widens_the_intensity_token_by_two() -> None:
    batch = _build_batch(("CCO",))
    baseline = _build_model(batch)
    widened = _build_model(
        batch, candidate_features=("c_cut_count", "c_size", "c_cut_adjacency")
    )

    assert widened.candidate_feature_dim - baseline.candidate_feature_dim == 2
    assert (
        widened.intensity_head.input_projection[0].in_features
        - baseline.intensity_head.input_projection[0].in_features
        == 2
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
        candidate_features=("c_cut_count", "c_size", "c_cut_bonds", "c_cut_adjacency"),
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


# --- the list mechanism itself ------------------------------------------------------------


def test_resolve_candidate_features_returns_registry_order() -> None:
    """Column order is the registry's, not the config's."""
    resolved = resolve_candidate_features(["c_charge", "c_mass", "c_size", "c_cut_count"])

    assert [spec.name for spec in resolved] == ["c_cut_count", "c_size", "c_mass", "c_charge"]


def test_resolve_candidate_features_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError) as exc_info:
        resolve_candidate_features(["c_ghost"])

    message = str(exc_info.value)
    assert "c_ghost" in message
    for name in CANDIDATE_FEATURES_BY_NAME:
        assert name in message


def test_resolve_candidate_features_rejects_a_duplicate() -> None:
    with pytest.raises(ValueError, match="duplicate candidate features"):
        resolve_candidate_features(["c_size", "c_size"])


def test_resolve_candidate_features_rejects_a_bare_string() -> None:
    """A bare string iterates into its characters, which is never what was meant."""
    with pytest.raises(ValueError, match="must be a list of names"):
        resolve_candidate_features("c_size")


def test_candidate_feature_width_sums_spec_widths() -> None:
    production = resolve_candidate_features(["c_cut_count", "c_size", "c_mass", "c_charge"])
    assert candidate_feature_width(production, edge_feature_dim=9) == 5

    all_features = resolve_candidate_features(list(CANDIDATE_FEATURES_BY_NAME))
    expected = 1 + 1 + 2 + 1 + COMPOSITION_FEATURE_DIM + 2 * 9 + 2 + (1 + COMPOSITION_FEATURE_DIM)
    assert candidate_feature_width(all_features, edge_feature_dim=9) == expected


def test_empty_candidate_feature_list_builds_no_block() -> None:
    """The empty list is a new state: no feature at all, not even the old unconditional two."""
    batch = _build_batch(("CCO",))
    model = _build_model(batch, candidate_features=())
    num_atoms = batch.x.shape[0]
    inputs = _candidate_inputs(
        model,
        batch,
        atom_indices=th.arange(num_atoms),
        atom_component_indices=th.zeros(num_atoms, dtype=th.long),
        component_batch_indices=th.tensor([0]),
        num_components=1,
        batch_size=1,
    )

    assert model.candidate_feature_specs == ()
    assert model.candidate_feature_dim == 0
    assert build_candidate_features((), inputs) is None
    assert candidate_feature_width((), edge_feature_dim=9) == 0


def test_build_candidate_features_concatenates_blocks_in_registry_order() -> None:
    """The combined block equals the per-feature blocks joined in registry order."""
    batch = _build_batch(("CCOC",))
    model = _build_model(
        batch, candidate_features=("c_charge", "c_mass", "c_size", "c_cut_count")
    )
    num_atoms = batch.x.shape[0]
    inputs = _candidate_inputs(
        model,
        batch,
        atom_indices=th.arange(num_atoms),
        atom_component_indices=th.zeros(num_atoms, dtype=th.long),
        component_batch_indices=th.tensor([0]),
        num_components=1,
        batch_size=1,
        cut_counts=th.tensor([2.0]),
    )

    assert [spec.name for spec in model.candidate_feature_specs] == [
        "c_cut_count",
        "c_size",
        "c_mass",
        "c_charge",
    ]
    combined = build_candidate_features(model.candidate_feature_specs, inputs)

    column = 0
    for spec in model.candidate_feature_specs:
        width = spec.width(inputs.edge_feature_dim)
        block = spec.build(inputs)
        assert th.equal(combined[:, column : column + width], block)
        column += width
    assert column == combined.shape[1]


def test_required_graph_fields_include_gasteiger_charges_only_with_c_charge() -> None:
    with_charge = resolve_candidate_features(["c_cut_count", "c_charge"])
    without_charge = resolve_candidate_features(["c_cut_count", "c_size"])

    assert candidate_feature_required_graph_fields(with_charge) == ("atom_gasteiger_charges",)
    assert candidate_feature_required_graph_fields(without_charge) == ()


def test_forward_does_not_require_gasteiger_charges_without_c_charge() -> None:
    """A model whose list lacks ``c_charge`` must not demand that graph field."""
    batch = _build_batch(("CCO",))
    model = _build_model(batch, candidate_features=("c_cut_count", "c_size"))
    del batch.atom_gasteiger_charges

    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0]),
        spec_nce=th.tensor([20.0]),
    )

    assert th.isfinite(output["pred_logprobs"]).all()


def test_forward_requires_gasteiger_charges_with_c_charge() -> None:
    batch = _build_batch(("CCO",))
    model = _build_model(batch, candidate_features=("c_cut_count", "c_size", "c_charge"))
    del batch.atom_gasteiger_charges

    with pytest.raises(ValueError, match="missing MASSter chemistry fields"):
        model(
            mol_pyg=batch,
            spec_prec_type=th.tensor([0]),
            spec_nce=th.tensor([20.0]),
        )


# --- candidate atom dropout --------------------------------------------------------------------


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


def test_candidate_atom_dropout_does_not_perturb_descriptor_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The correctness trap of the refactor: dropout perturbs the pooled embeddings only.

    ``c_size``/``c_mass`` are read off the TRUE atom membership regardless of
    ``candidate_atom_dropout``, so the descriptor block built for the same batch and seed must be
    bit-identical whether dropout is 0.0 or 0.5.
    """
    batch = _build_batch(("c1ccccc1CCO", "CCO"))
    captured: list[th.Tensor] = []
    real_build_candidate_features = masster_model_module.build_candidate_features

    def _spy(specs, inputs):
        block = real_build_candidate_features(specs, inputs)
        captured.append(block)
        return block

    monkeypatch.setattr(masster_model_module, "build_candidate_features", _spy)

    def _run(dropout: float) -> th.Tensor:
        captured.clear()
        model = _build_model(
            batch,
            candidate_features=("c_cut_count", "c_size", "c_mass"),
            candidate_atom_dropout=dropout,
        )
        model.train()
        th.manual_seed(41)
        model(
            mol_pyg=batch,
            spec_prec_type=th.tensor([0, 0]),
            spec_nce=th.tensor([20.0, 20.0]),
        )
        assert len(captured) == 1, "expected exactly one descriptor block per forward"
        return captured[0]

    no_dropout = _run(0.0)
    with_dropout = _run(0.5)

    assert th.equal(no_dropout, with_dropout)


# --- candidate boundary dropout ----------------------------------------------------------------


def test_candidate_boundary_dropout_is_inactive_outside_training() -> None:
    """Evaluation feeds the true boundary block, so validation metrics stay deterministic."""
    batch = _build_batch(("CCO",))
    model = _build_model(batch, candidate_boundary_dropout=0.5)
    model.eval()
    boundary = th.randn(7, 5)

    assert th.equal(model._candidate_boundary_dropout(boundary), boundary)


def test_candidate_boundary_dropout_is_inactive_at_probability_zero() -> None:
    """The default returns the input and draws no random numbers, so old runs are bit-identical."""
    batch = _build_batch(("CCO",))
    model = _build_model(batch)
    model.train()
    boundary = th.randn(7, 5)
    rng_before = th.get_rng_state()

    output = model._candidate_boundary_dropout(boundary)

    assert th.equal(output, boundary)
    assert th.equal(th.get_rng_state(), rng_before)


def test_candidate_boundary_dropout_drops_whole_blocks_with_inverted_scaling() -> None:
    """Each row is all zero or the input times 1/(1-p), and about p of the rows are dropped."""
    batch = _build_batch(("CCO",))
    model = _build_model(batch, candidate_boundary_dropout=0.25)
    model.train()
    th.manual_seed(11)
    boundary = th.rand(2000, 6) + 0.5  # strictly positive, so a zero entry means dropped

    output = model._candidate_boundary_dropout(boundary)

    assert output.shape == boundary.shape
    dropped = (output == 0).all(dim=1)
    kept = ~dropped
    assert not ((output == 0).any(dim=1) & kept).any(), "a block is dropped whole or not at all"
    assert th.allclose(output[kept], boundary[kept] / 0.75)
    assert dropped.float().mean().item() == pytest.approx(0.25, abs=0.03)
    # Inverted scaling keeps the expected block, which is what the token's linear layer sees.
    assert output.mean().item() == pytest.approx(boundary.mean().item(), rel=0.05)


def test_candidate_boundary_dropout_rejects_certain_dropout() -> None:
    batch = _build_batch(("CCO",))
    with pytest.raises(ValueError, match="candidate_boundary_dropout must be in"):
        _build_model(batch, candidate_boundary_dropout=1.0)


@pytest.mark.parametrize("unique_fragment_mode", [False, True])
def test_candidate_boundary_dropout_reaches_only_the_intensity_token(
    unique_fragment_mode: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In training the head's boundary input is the dropped block; at eval it is untouched.

    Also pins that the rate adds no parameters: at eval a dropout model and a default model built
    from the same seed give identical predictions.
    """
    batch = _build_batch(("CCOC", "CCO"))
    kwargs = {
        "max_masks": 4,
        "max_enumerated_masks": 8,
        "unique_fragment_mode": unique_fragment_mode,
    }
    captured: list[tuple[th.Tensor, ...]] = []

    def _forward(model: MassterModel) -> dict[str, th.Tensor]:
        return model(
            mol_pyg=batch,
            spec_prec_type=th.tensor([0, 0]),
            spec_nce=th.tensor([10.0, 40.0]),
        )

    dropout_model = _build_model(batch, candidate_boundary_dropout=0.5, **kwargs)
    default_model = _build_model(batch, **kwargs)
    real_head_forward = dropout_model.intensity_head.forward

    def _spy(fragment, complement, boundary, *args, **head_kwargs):
        captured.append((fragment, complement, boundary))
        return real_head_forward(fragment, complement, boundary, *args, **head_kwargs)

    monkeypatch.setattr(dropout_model.intensity_head, "forward", _spy)

    dropout_model.eval()
    default_model.eval()
    eval_output = _forward(dropout_model)
    assert th.allclose(eval_output["pred_logprobs"], _forward(default_model)["pred_logprobs"])
    eval_boundary = captured[-1][2]

    dropout_model.train()
    th.manual_seed(5)
    train_output = _forward(dropout_model)
    train_boundary = captured[-1][2]

    assert train_boundary.shape == eval_boundary.shape
    dropped = (train_boundary == 0).all(dim=1) & (eval_boundary != 0).any(dim=1)
    kept = ~(train_boundary == 0).all(dim=1)
    assert dropped.any(), "the fixture must actually drop a block"
    assert th.allclose(train_boundary[kept], 2.0 * eval_boundary[kept], atol=1e-6)
    assert th.isfinite(train_output["pred_logprobs"]).all()
    train_output["pred_logprobs"].mean().backward()
    assert dropout_model.intensity_head.input_projection[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("unique_fragment_mode", [False, True])
def test_forward_runs_with_an_empty_candidate_feature_list(unique_fragment_mode: bool) -> None:
    """A token with no descriptor block at all is new, and has to survive the model forward."""
    batch = _build_batch(("CCOC", "CCO"))
    model = _build_model(
        batch,
        max_masks=4,
        max_enumerated_masks=8,
        unique_fragment_mode=unique_fragment_mode,
        candidate_features=(),
    )

    assert model.candidate_feature_specs == ()
    assert model.candidate_feature_dim == 0
    assert model.intensity_head.candidate_feature_dim == 0

    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0, 0]),
        spec_nce=th.tensor([10.0, 40.0]),
    )

    assert th.isfinite(output["pred_logprobs"]).all()
    output["pred_logprobs"].mean().backward()


def test_describe_candidate_features_reports_the_resolved_layout() -> None:
    """The line `masster_pl` logs must name every column group and the total."""
    specs = resolve_candidate_features(("c_cut_count", "c_size", "c_mass", "c_charge"))

    assert (
        describe_candidate_features(specs, 9)
        == "c_cut_count:1, c_size:1, c_mass:2, c_charge:1 (5 columns)"
    )
    assert describe_candidate_features(resolve_candidate_features(()), 9) == (
        "no candidate features (0 columns)"
    )
    # The cut-bond width follows the precursor bond features, so the same list describes
    # differently under the ring-topology arm's wider bond set.
    assert "c_cut_bonds:32" in describe_candidate_features(
        resolve_candidate_features(("c_cut_bonds",)), 16
    )


# --- c_heavy_side and the light-side token ------------------------------------------------------


def test_heavy_side_flags_the_lighter_side_and_counts_the_heavier() -> None:
    """Ethanol C2H6O (46 Da): the precursor, a light CH3 and a heavy C2H5 fragment."""
    batch = _build_batch(("CCO",))
    model = _build_model(batch, candidate_features=("c_cut_count", "c_size", "c_heavy_side"))
    carbon_column = CANONICAL_ELEMENT_ORDER.index("C")
    hydrogen_column = CANONICAL_ELEMENT_ORDER.index("H")
    oxygen_column = CANONICAL_ELEMENT_ORDER.index("O")

    # Component 0 is the intact molecule, 1 the terminal carbon, 2 the two carbons.
    inputs = _candidate_inputs(
        model,
        batch,
        atom_indices=th.tensor([0, 1, 2, 0, 0, 1]),
        atom_component_indices=th.tensor([0, 0, 0, 1, 2, 2]),
        component_batch_indices=th.tensor([0, 0, 0]),
        num_components=3,
        batch_size=1,
    )
    block = CANDIDATE_FEATURES_BY_NAME["c_heavy_side"].build(inputs)

    assert block.shape == (3, 1 + COMPOSITION_FEATURE_DIM)
    assert block[:, 0].tolist() == [0.0, 1.0, 0.0]
    counts = th.expm1(block[:, 1:]).round().long()
    # The precursor is its own heavier side.
    assert (counts[0, carbon_column], counts[0, oxygen_column], counts[0, hydrogen_column]) == (
        2,
        1,
        6,
    )
    # CH3 (15 Da) is the lighter side, so the heavier side is its complement CH3O.
    assert (counts[1, carbon_column], counts[1, oxygen_column], counts[1, hydrogen_column]) == (
        1,
        1,
        3,
    )
    # C2H5 (29 Da) is the heavier side itself.
    assert (counts[2, carbon_column], counts[2, oxygen_column], counts[2, hydrogen_column]) == (
        2,
        0,
        5,
    )


def test_light_side_token_pools_only_the_lighter_side() -> None:
    """Same weights and candidates; only the pooled slots of the token change."""
    batch = _build_batch(("CCOC(=O)c1ccccc1", "CCO"))
    features = ("c_cut_count", "c_size", "c_mass", "c_heavy_side")
    model = _build_model(
        batch,
        max_masks=6,
        max_enumerated_masks=16,
        unique_fragment_mode=True,
        candidate_features=features,
    ).eval()
    seen: list[tuple[th.Tensor, th.Tensor, th.Tensor]] = []
    real_forward = model.intensity_head.forward

    def _spy(*args, **kwargs):
        seen.append((args[0], args[1], kwargs["candidate_features"]))
        return real_forward(*args, **kwargs)

    model.intensity_head.forward = _spy

    def _forward() -> dict[str, th.Tensor]:
        return model(
            mol_pyg=batch, spec_prec_type=th.tensor([0, 0]), spec_nce=th.tensor([20.0, 40.0])
        )

    with th.no_grad():
        clean = _forward()
        model.light_side_token = True
        light = _forward()

    fragment, complement, descriptors = seen[0]
    light_fragment, light_complement, light_descriptors = seen[1]
    # Columns: c_cut_count, c_size, c_mass (2), then c_heavy_side's lighter-side flag.
    lighter = descriptors[:, 4].bool()
    precursor = descriptors[:, 3] > 1.0 - 1e-4
    assert lighter.any() and (~lighter & ~precursor).any()
    assert th.equal(light_fragment, th.where(lighter.unsqueeze(1), fragment, complement))
    assert th.count_nonzero(light_complement) == 0
    assert th.count_nonzero(light_fragment[precursor]) == 0
    assert th.equal(light_descriptors, descriptors)
    assert th.equal(light["pred_mzs"], clean["pred_mzs"])
    assert not th.allclose(light["pred_logprobs"], clean["pred_logprobs"])


def test_light_side_token_requires_unique_fragments_and_the_heavy_side_descriptor() -> None:
    batch = _build_batch(("CCO",))

    with pytest.raises(ValueError, match="requires unique_fragment_mode"):
        _build_model(
            batch, candidate_features=("c_cut_count", "c_heavy_side"), light_side_token=True
        )
    with pytest.raises(ValueError, match="requires c_heavy_side"):
        _build_model(batch, unique_fragment_mode=True, light_side_token=True)
