"""MASSter's dataset filter: drop molecule/adduct pairs with no chemically valid ion channel.

``MassterHooks.filter_dataset`` runs it inside ``SpecMolDataset.__init__``, after the frames are
built and before preprocessing; ``BaseDataset.keep_spectra`` applies the verdicts.
"""

import logging
from collections import Counter

import pandas as pd
import rdkit.Chem as Chem

from fragnnet.utils.formula_utils import (
    H_MASS,
    PERIODIC_TABLE,
    PREC_TYPE_TO_CMF_COMP_DIFF,
    PREC_TYPE_TO_CMF_MASS_DIFF,
    PREC_TYPE_TO_COMP_DIFF,
    PREC_TYPE_TO_MASS_DIFF,
    cmf_duplicate_crf_h_shift,
)


def masster_has_valid_ion(
    mol: Chem.Mol,
    precursor_type: str,
    max_h_transfer: int,
    include_cmf: bool,
    h_transfer_bound: str = "none",
) -> bool:
    """Check whether an uncut component can supply any configured ion channel.

    Counts and masses match MASSter's graph chemistry. If a connected component
    cannot supply the removed elements, none of its smaller fragments can either.
    This check runs on CPU before spectra enter a training/evaluation batch.

    When ``h_transfer_bound`` is not ``"none"``, hydrogen transfer draws from the
    complementary neutral fragment, so a component can gain at most the hydrogens the rest of
    the molecule holds -- none for a single-component molecule. This keeps the filter's notion
    of reachability aligned with ``MassterModel``'s; leave it at the model's own setting.
    Only hydrogen-free molecules change verdict (2 of 35,025 in NIST20), and correctly so: a
    molecule with no hydrogen cannot form ``[M-H]-``.

    Args:
        mol: Molecule to test.
        precursor_type: Adduct string.
        max_h_transfer: Width of the hydrogen-transfer ladder.
        include_cmf: Whether charge-migration channels are enabled.
        h_transfer_bound: Matching ``masster_h_transfer_bound`` setting. The valence
            term needs per-fragment cut counts and so does not apply to this whole-component
            admissibility check; ``"valence"`` therefore behaves as ``"complement"`` here.
    """
    channels = [
        (PREC_TYPE_TO_COMP_DIFF[precursor_type], PREC_TYPE_TO_MASS_DIFF[precursor_type], shift)
        for shift in range(-max_h_transfer, max_h_transfer + 1)
    ]
    if (
        include_cmf
        and precursor_type in PREC_TYPE_TO_CMF_COMP_DIFF
        and cmf_duplicate_crf_h_shift(precursor_type, max_h_transfer) is None
    ):
        channels.extend(
            (
                PREC_TYPE_TO_CMF_COMP_DIFF[precursor_type],
                PREC_TYPE_TO_CMF_MASS_DIFF[precursor_type],
                shift,
            )
            for shift in range(-max_h_transfer, max_h_transfer + 1)
        )
    molecule_hydrogens = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms()) + sum(
        atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()
    )
    for component in Chem.GetMolFrags(mol, asMols=True):
        counts = Counter(atom.GetSymbol() for atom in component.GetAtoms())
        counts["H"] += sum(atom.GetTotalNumHs() for atom in component.GetAtoms())
        mass = sum(
            PERIODIC_TABLE.GetMostCommonIsotopeMass(element) * count
            for element, count in counts.items()
        )
        complement_hydrogens = molecule_hydrogens - counts["H"]
        for delta, mass_delta, shift in channels:
            if h_transfer_bound != "none" and shift > complement_hydrogens:
                continue
            if (
                counts["H"] + delta.get("H", 0) + shift >= 0
                and all(
                    counts[element] + count >= 0
                    for element, count in delta.items()
                    if element != "H"
                )
                and mass + mass_delta + shift * H_MASS > 0.0
            ):
                return True
    return False


def filter_masster_spectra(
    dataset,
    max_h_transfer: int,
    include_cmf: bool,
    h_transfer_bound: str = "none",
) -> None:
    """Drop incompatible molecule/adduct pairs before preprocessing and batching.

    Only the verdicts are MASSter's; ``BaseDataset.keep_spectra`` drops the rows and rebuilds
    the dataset's derived state.

    Args:
        dataset: A ``SpecMolDataset`` after ``_base_init``, before preprocessing.
        max_h_transfer: Largest hydrogen shift a channel may carry.
        include_cmf: Whether charge-migration channels count.
        h_transfer_bound: The model's hydrogen-transfer bound (``none`` ignores it).

    Raises:
        ValueError: If no spectrum is left.
    """
    valid_pairs = {
        (mol_id, prec_type): masster_has_valid_ion(
            dataset.mol_df.loc[mol_id, "mol"],
            prec_type,
            max_h_transfer,
            include_cmf,
            h_transfer_bound,
        )
        for mol_id, prec_type in dataset.spec_df[["mol_id", "prec_type"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    if valid_pairs and not any(valid_pairs.values()):
        raise ValueError("MASSter dataset has no chemically valid spectra after filtering")

    def keep(frame: pd.DataFrame) -> pd.Series:
        return pd.Series(
            [
                valid_pairs.get(pair, False)
                for pair in frame[["mol_id", "prec_type"]].itertuples(index=False, name=None)
            ],
            index=frame.index,
            dtype=bool,
        )

    removed = dataset.keep_spectra(keep)
    if removed:
        logging.getLogger(__name__).warning(
            "MASSter filtered %d/%d spectra with no chemically valid ion channels",
            removed,
            removed + len(dataset.spec_df),
        )
