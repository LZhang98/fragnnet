"""Per-candidate descriptor columns for MASSter's intensity token.

The candidate token that :class:`CandidateIntensityTransformerHead` projects is three pooled
embeddings (fragment, complement, boundary) plus the spectrum context, plus a block of scalar
descriptors of the candidate itself. This module owns that descriptor block: one spec per
column group, resolved from the ``masster_candidate_features`` config list.

Two parts of the token are deliberately NOT governed by that list:

* the pooled embeddings and the context vector, which are the token's backbone rather than
  descriptors — dropping one is an architecture change, not an ablation;
* the proposal-derived columns (the per-mask component count and the base charge/mask
  log-probabilities), whose presence follows ``masster_proposal_anchored`` and the
  unique-fragment mode rather than being chosen. Listing them would admit the incoherent
  configuration "anchored *and* base-probability column".

Anything that parameterizes a module keeps its own scalar config key on that basis:
``masster_boundary_bond_encoder`` selects how the boundary block is computed and
``masster_candidate_atom_dropout`` regularizes the pooled embeddings; neither owns a
descriptor column.

Columns are concatenated in :data:`CANDIDATE_FEATURE_SPECS` order, not in the order the config
lists them, so two configs naming the same features build identical tokens. That order is the
order the head appended these columns before they became a list, so the descriptor block itself
is unchanged.

The block's position inside the token did move, and in one configuration that is visible. The
head now appends the whole block last. Before, the cut count and the atom count sat immediately
after the context and *ahead* of the proposal-derived columns (the per-mask component count and
the base charge/mask log-probabilities), while every other descriptor sat after them. So under
``masster_unique_fragment_mode: False`` or ``masster_proposal_anchored: False`` --
both ``config/template.yml`` defaults, inherited by ``config/debug/debug_tiny_masster.yml``
and ``config/nist20_merged_inchikey/masster/s0.yml`` -- those two columns now follow the
proposal columns instead of preceding them. Concatenating into one ``nn.Linear`` makes that a
permutation of weight columns: it changes which weight a given seed initializes and nothing
else, and no checkpoint predates it. Every ``ma_mi`` arm pins both keys True and is unaffected.

Every builder emits its final float columns itself, in ``inputs.dtype``. The cast is part of the
column's definition rather than a step the head adds afterwards: ``c_size`` casts *then* takes
``log1p`` while ``c_mass`` takes ``log1p`` in the source float64 and casts afterwards, and those
two orders are not interchangeable in reduced precision.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property

import torch as th
from torch_geometric.data import Batch

from fragnnet.model.bond_fragment_heads import (
    _validate_tensor_condition,
    boundary_bond_endpoint_orientation,
    pool_boundary_bond_embeddings,
)
from fragnnet.utils.formula_utils import H_MASS
from fragnnet.utils.frag_utils import CANONICAL_ELEMENT_ORDER, CANONICAL_H_IDX

# Element vocabulary for composition features: the canonical order plus one shared column for
# anything outside it, matching the unknown bucket ``a_onehot`` gives unlisted elements.
COMPOSITION_FEATURE_DIM = len(CANONICAL_ELEMENT_ORDER) + 1

# Hydrogen's column inside that vocabulary.
COMPOSITION_HYDROGEN_COLUMN = CANONICAL_H_IDX


def per_atom_neutral_masses(mol_pyg: Batch) -> th.Tensor:
    """Return each atom's monoisotopic mass including its implicit hydrogens."""
    return mol_pyg.atom_monoisotopic_masses + (
        mol_pyg.atom_hydrogen_counts.to(mol_pyg.atom_monoisotopic_masses) * H_MASS
    )


def component_neutral_masses(
    mol_pyg: Batch,
    atom_indices: th.Tensor,
    atom_component_indices: th.Tensor,
    num_components: int,
) -> th.Tensor:
    """Sum neutral elemental masses over each component's atoms.

    Args:
        mol_pyg: Batched precursor graphs carrying the chemistry fields.
        atom_indices: Atom index of each membership entry.
        atom_component_indices: Component index of each membership entry.
        num_components: Number of components.

    Returns:
        Neutral fragment mass per component, shape ``(num_components,)``.
    """
    per_atom = per_atom_neutral_masses(mol_pyg)
    return per_atom.new_zeros(num_components).index_add(
        0,
        atom_component_indices,
        per_atom[atom_indices],
    )


def precursor_neutral_masses(mol_pyg: Batch, batch_size: int) -> th.Tensor:
    """Return the intact neutral mass of every precursor in the batch."""
    per_atom = per_atom_neutral_masses(mol_pyg)
    return per_atom.new_zeros(batch_size).index_add(0, mol_pyg.batch, per_atom)


def element_counts_by_group(
    mol_pyg: Batch,
    atom_indices: th.Tensor,
    group_indices: th.Tensor,
    num_groups: int,
    element_columns: th.Tensor,
    num_columns: int,
    hydrogen_column: int,
) -> th.Tensor:
    """Count elements per group of atoms, hydrogens included.

    Groups are components when called with a component-membership relation and molecules
    when called with ``mol_pyg.batch``, so the same reduction produces both a fragment
    composition and its precursor composition.

    Hydrogen is counted twice over, exactly as the channel-validity routine counts it: an
    explicit H atom contributes through its own atomic number, and every heavy atom
    contributes its implicit hydrogen count.

    Args:
        mol_pyg: Batched precursor graphs carrying the per-atom chemistry fields.
        atom_indices: Atom index of each membership entry.
        group_indices: Group index of each membership entry.
        num_groups: Number of groups.
        element_columns: Lookup from atomic number to column, sized to cover every atomic
            number the graphs can carry; unknown elements share the final column.
        num_columns: Number of element columns, one past the largest value in
            ``element_columns``. Passed rather than read off the tensor, which would cost a
            device synchronization on every batch.
        hydrogen_column: Column holding hydrogen counts.

    Returns:
        Element counts of shape ``(num_groups, num_columns)``.
    """
    atomic_numbers = mol_pyg.atom_atomic_numbers[atom_indices]
    membership_columns = element_columns[atomic_numbers]
    counts = th.zeros(
        num_groups * num_columns,
        dtype=th.long,
        device=atom_indices.device,
    )
    counts.index_add_(
        0,
        group_indices * num_columns + membership_columns,
        th.ones_like(membership_columns),
    )
    counts.index_add_(
        0,
        group_indices * num_columns + hydrogen_column,
        mol_pyg.atom_hydrogen_counts[atom_indices].long(),
    )
    return counts.view(num_groups, num_columns)


@dataclass(frozen=True)
class CandidateFeatureInputs:
    """Everything a candidate feature builder may read, for one forward branch.

    The two forward branches of :class:`MassterModel` name their tensors differently but
    carry the same relations, so each builds one of these and the builders stay branch-blind.

    ``atom_indices`` and ``atom_component_indices`` must be the TRUE membership, never the
    subset ``_candidate_atom_dropout`` returns. That dropout perturbs the pooled component
    embeddings only; masses, sizes, charges and compositions keep the real atom set, and
    feeding the dropped-out relation here would make them vary run to run during training.

    Derived quantities are cached rather than recomputed: the neutral masses are read both by
    ``c_mass`` and by the head's pairwise mass bias, which is enabled independently of whether
    ``c_mass`` is in the feature list.

    Attributes:
        mol_pyg: Batched precursor graphs.
        atom_indices: Atom index of each component-membership entry.
        atom_component_indices: Component index of each membership entry.
        boundary_bond_indices: Bond index of each (bond, component) boundary pair.
        boundary_component_indices: Component index of each boundary pair.
        component_batch_indices: Spectrum index per component.
        cut_counts: Number of severed bonds per component.
        composition_element_columns: Atomic number to element column lookup.
        num_components: Number of components.
        batch_size: Number of spectra in the batch.
        dtype: Dtype every built column is emitted in, the token's own dtype.
    """

    mol_pyg: Batch
    atom_indices: th.Tensor
    atom_component_indices: th.Tensor
    boundary_bond_indices: th.Tensor
    boundary_component_indices: th.Tensor
    component_batch_indices: th.Tensor
    cut_counts: th.Tensor
    composition_element_columns: th.Tensor
    num_components: int
    batch_size: int
    dtype: th.dtype

    @cached_property
    def edge_feature_dim(self) -> int:
        """Width of the precursor bond features, which sizes ``c_cut_bonds``."""
        return int(self.mol_pyg.chemical_bond_attr.shape[1])

    @cached_property
    def component_sizes(self) -> th.Tensor:
        """Atom count per component, ``th.long``."""
        return th.bincount(self.atom_component_indices, minlength=self.num_components)

    @cached_property
    def component_neutral_masses(self) -> th.Tensor:
        """Neutral fragment mass per component, in the graph's mass dtype."""
        return component_neutral_masses(
            self.mol_pyg,
            self.atom_indices,
            self.atom_component_indices,
            self.num_components,
        )

    @cached_property
    def precursor_neutral_masses(self) -> th.Tensor:
        """Intact neutral mass per spectrum, in the graph's mass dtype."""
        masses = precursor_neutral_masses(self.mol_pyg, self.batch_size)
        _validate_tensor_condition(
            (masses > 0).all(),
            "every precursor must have a positive neutral mass",
        )
        return masses

    @cached_property
    def complement_element_counts(self) -> th.Tensor:
        """Element counts of each candidate's complementary neutral loss.

        The complement is the precursor minus the candidate, so both sides are counted with
        the same reduction and the difference is exact by construction.
        """
        component_counts = element_counts_by_group(
            self.mol_pyg,
            self.atom_indices,
            self.atom_component_indices,
            self.num_components,
            self.composition_element_columns,
            COMPOSITION_FEATURE_DIM,
            COMPOSITION_HYDROGEN_COLUMN,
        )
        precursor_counts = element_counts_by_group(
            self.mol_pyg,
            th.arange(self.mol_pyg.batch.shape[0], device=self.mol_pyg.batch.device),
            self.mol_pyg.batch,
            self.batch_size,
            self.composition_element_columns,
            COMPOSITION_FEATURE_DIM,
            COMPOSITION_HYDROGEN_COLUMN,
        )
        complement_counts = precursor_counts[self.component_batch_indices] - component_counts
        _validate_tensor_condition(
            (complement_counts >= 0).all(),
            "a candidate holds more atoms of some element than its precursor",
        )
        return complement_counts


def _build_cut_count(inputs: CandidateFeatureInputs) -> th.Tensor:
    """Number of bonds severed to make the candidate, as a raw count."""
    return inputs.cut_counts.to(inputs.dtype).unsqueeze(1)


def _build_size(inputs: CandidateFeatureInputs) -> th.Tensor:
    """``log1p`` of the candidate's atom count, cast before the transform."""
    return th.log1p(inputs.component_sizes.to(inputs.dtype)).unsqueeze(1)


def _build_mass(inputs: CandidateFeatureInputs) -> th.Tensor:
    """``log1p`` neutral mass and the fraction of the intact precursor mass.

    Without these no mass, m/z or composition reaches a learned parameter at all: the rest of
    the token is pooled embeddings plus counts.
    """
    masses = inputs.component_neutral_masses
    precursors = inputs.precursor_neutral_masses
    return th.stack(
        (
            th.log1p(masses),
            masses / precursors[inputs.component_batch_indices],
        ),
        dim=1,
    ).to(inputs.dtype)


def _build_charge(inputs: CandidateFeatureInputs) -> th.Tensor:
    """Sum of the candidate's Gasteiger partial charges, FraGNNet's ``frag_q``.

    Independent of the ``a_charge`` atom feature: ``atom_gasteiger_charges`` is attached to
    every molecular graph.
    """
    atom_charges = inputs.mol_pyg.atom_gasteiger_charges
    summed = atom_charges.new_zeros(inputs.num_components).index_add(
        0,
        inputs.atom_component_indices,
        atom_charges[inputs.atom_indices],
    )
    return summed.to(inputs.dtype).unsqueeze(1)


def _build_nl_composition(inputs: CandidateFeatureInputs) -> th.Tensor:
    """``log1p`` element counts of the complementary neutral loss.

    GLACIER's ``psi(r_form - f_k)`` term. The head already sees a mean-pooled embedding of the
    complement's atoms, but a mean of learned vectors carries no exact counts, and ``c_mass``
    gives the complement's mass only -- neither separates a CO loss (27.9949) from a C2H4 loss
    (28.0313).
    """
    counts = inputs.complement_element_counts
    return th.log1p(counts.to(inputs.mol_pyg.atom_monoisotopic_masses)).to(inputs.dtype)


def _build_cut_bonds(inputs: CandidateFeatureInputs) -> th.Tensor:
    """Summed then averaged input features of the severed bonds.

    The boundary embedding reaches the intensity head through atom endpoints only -- it never
    reads ``chemical_bond_attr`` -- and an internal atom mixes all of its incident bonds into
    one state, so for an endpoint of degree > 1 the cut bond's own attributes are not cleanly
    recoverable. The summed half is the multiset of broken bond types, which also delivers how
    many cut bonds were ring bonds.

    A candidate with no severed bonds -- the intact precursor -- scores zero in both halves,
    which is the correct "nothing broken".
    """
    bond_features = inputs.mol_pyg.chemical_bond_attr
    pooled = [
        pool_boundary_bond_embeddings(
            bond_features,
            inputs.boundary_bond_indices,
            inputs.boundary_component_indices,
            num_components=inputs.num_components,
            reduction=reduction,
        )
        for reduction in ("sum", "mean")
    ]
    return th.cat(pooled, dim=1).to(inputs.dtype)


def _build_cut_adjacency(inputs: CandidateFeatureInputs) -> th.Tensor:
    """Cut-bond pairs meeting at a shared atom, split by retained then lost side.

    Two cut bonds sharing an atom is a relation *between* cuts, so mean pooling over per-bond
    encodings cannot express it and no bond feature carries it either. The side matters
    chemically: a shared retained atom means one atom of the fragment lost two substituents,
    while a shared lost atom means the fragment expelled a bridging atom -- the CO/CH2/S
    extrusion shape.

    Two bonds share at most one atom in a simple graph, so summing ``C(count, 2)`` over the
    (component, atom) incidence counts each adjacent pair exactly once.
    """
    num_atoms = int(inputs.mol_pyg.batch.shape[0])
    inside_atoms, outside_atoms = boundary_bond_endpoint_orientation(
        inputs.mol_pyg.chemical_bond_index,
        inputs.boundary_bond_indices,
        inputs.boundary_component_indices,
        inputs.atom_indices,
        inputs.atom_component_indices,
        num_atoms=num_atoms,
    )
    sides = []
    for side_atoms in (inside_atoms, outside_atoms):
        incidence_keys = inputs.boundary_component_indices * num_atoms + side_atoms
        unique_keys, counts = th.unique(incidence_keys, return_counts=True)
        adjacent_pairs = counts * (counts - 1) // 2
        sides.append(
            th.zeros(
                inputs.num_components,
                dtype=adjacent_pairs.dtype,
                device=adjacent_pairs.device,
            ).index_add(
                0,
                th.div(unique_keys, num_atoms, rounding_mode="floor"),
                adjacent_pairs,
            )
        )
    return th.stack(sides, dim=1).to(inputs.dtype)


@dataclass(frozen=True)
class CandidateFeatureSpec:
    """One named descriptor group inside the candidate token.

    Attributes:
        name: The name a config lists.
        width: Columns this feature contributes, given the precursor bond-feature width.
        build: Builds the ``(num_components, width)`` block in ``inputs.dtype``.
        requires: Graph fields the builder reads that a stale molecular cache may lack.
    """

    name: str
    width: Callable[[int], int]
    build: Callable[[CandidateFeatureInputs], th.Tensor]
    requires: tuple[str, ...] = ()


# Canonical column order. Adding a descriptor is one entry here plus one config-list entry.
CANDIDATE_FEATURE_SPECS: tuple[CandidateFeatureSpec, ...] = (
    CandidateFeatureSpec(
        name="c_cut_count",
        width=lambda edge_feature_dim: 1,
        build=_build_cut_count,
    ),
    CandidateFeatureSpec(
        name="c_size",
        width=lambda edge_feature_dim: 1,
        build=_build_size,
    ),
    CandidateFeatureSpec(
        name="c_mass",
        width=lambda edge_feature_dim: 2,
        build=_build_mass,
    ),
    CandidateFeatureSpec(
        name="c_charge",
        width=lambda edge_feature_dim: 1,
        build=_build_charge,
        requires=("atom_gasteiger_charges",),
    ),
    CandidateFeatureSpec(
        name="c_nl_composition",
        width=lambda edge_feature_dim: COMPOSITION_FEATURE_DIM,
        build=_build_nl_composition,
    ),
    CandidateFeatureSpec(
        name="c_cut_bonds",
        width=lambda edge_feature_dim: 2 * edge_feature_dim,
        build=_build_cut_bonds,
        requires=("chemical_bond_attr",),
    ),
    CandidateFeatureSpec(
        name="c_cut_adjacency",
        width=lambda edge_feature_dim: 2,
        build=_build_cut_adjacency,
    ),
)

CANDIDATE_FEATURES_BY_NAME = {spec.name: spec for spec in CANDIDATE_FEATURE_SPECS}


def resolve_candidate_features(names: Sequence[str]) -> tuple[CandidateFeatureSpec, ...]:
    """Resolve configured feature names to specs in canonical order.

    The config lists a set; the order columns are concatenated in is this module's, so two
    configs naming the same features build identical tokens.

    Args:
        names: Feature names from ``masster_candidate_features``.

    Returns:
        The matching specs, in :data:`CANDIDATE_FEATURE_SPECS` order.

    Raises:
        ValueError: If ``names`` is a bare string, holds a duplicate, or names a feature that
            does not exist.
    """
    if isinstance(names, str):
        raise ValueError(
            "masster_candidate_features must be a list of names, not a single string"
        )
    requested = list(names)
    duplicates = sorted({name for name in requested if requested.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate candidate features: {duplicates}")
    unknown = sorted(set(requested) - set(CANDIDATE_FEATURES_BY_NAME))
    if unknown:
        raise ValueError(
            f"unknown candidate features: {unknown}; "
            f"known features: {sorted(CANDIDATE_FEATURES_BY_NAME)}"
        )
    selected = set(requested)
    return tuple(spec for spec in CANDIDATE_FEATURE_SPECS if spec.name in selected)


def candidate_feature_width(
    specs: Sequence[CandidateFeatureSpec],
    edge_feature_dim: int,
) -> int:
    """Return the total column count the intensity head must project from."""
    return sum(spec.width(edge_feature_dim) for spec in specs)


def candidate_feature_required_graph_fields(
    specs: Sequence[CandidateFeatureSpec],
) -> tuple[str, ...]:
    """Return the graph fields the resolved features read, deduplicated in spec order."""
    fields: list[str] = []
    for spec in specs:
        fields.extend(field for field in spec.requires if field not in fields)
    return tuple(fields)


def build_candidate_features(
    specs: Sequence[CandidateFeatureSpec],
    inputs: CandidateFeatureInputs,
) -> th.Tensor | None:
    """Build the descriptor block for one batch of candidates.

    Args:
        specs: Resolved specs, in canonical order.
        inputs: The tensors every builder reads, built from the TRUE atom membership.

    Returns:
        ``(num_components, total_width)`` in ``inputs.dtype``, or ``None`` when no feature is
        configured, which leaves the token without a descriptor block at all.

    Raises:
        ValueError: If a builder returns a block of the wrong shape.
    """
    if not specs:
        return None
    blocks = []
    for spec in specs:
        block = spec.build(inputs)
        expected = (inputs.num_components, spec.width(inputs.edge_feature_dim))
        if block.shape != expected:
            raise ValueError(
                f"candidate feature {spec.name} built {tuple(block.shape)}, expected {expected}"
            )
        blocks.append(block)
    return th.cat(blocks, dim=1)


def describe_candidate_features(
    specs: Sequence[CandidateFeatureSpec],
    edge_feature_dim: int,
) -> str:
    """Return a one-line description of the resolved token layout, for the run log."""
    if not specs:
        return "no candidate features (0 columns)"
    parts = ", ".join(f"{spec.name}:{spec.width(edge_feature_dim)}" for spec in specs)
    return f"{parts} ({candidate_feature_width(specs, edge_feature_dim)} columns)"
