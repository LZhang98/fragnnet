"""Molecule-level training exposure for the rare chemistry sampling arms.

Implements arms A1 and A3 of ``docs/plans/masster/rare_chemistry_augmentation_plan.md``.
Each training draw picks a molecule, then one of its recorded acquisition keys uniformly,
then one real spectrum with that key uniformly. A1 picks molecules uniformly. A3 picks them
from a capped mixture that upweights molecules carrying functional-group classes with few
training molecules. Every count here is taken from the training dataset after its filters.

Molecule identity is the 2D InChIKey block (``inchikey_s``), the NIST20 split key. The
acquisition key is ``(adduct, instrument, fragmentation mode, recorded CE signature)``. The CE
signature is read from the raw spectrum table because the dataset imputes missing NCE from ACE
and vice versa; imputed values must not merge differently recorded acquisitions.
"""

import logging

import numpy as np
import pandas as pd
import torch as th

from fragnnet.dataset.data_sampler import MoleculeAcquisitionSampler
from fragnnet.utils.data_utils import rdkit_import

logger = logging.getLogger(__name__)

# Frozen functional-group panel, version 1 (2026-10-10). Never edit an entry in place: add a
# new versioned panel instead, so rarity weights and frozen evaluation cohorts stay comparable.
# Classes overlap; a molecule carries every class it matches. S=O and P=O/P=S groups match both
# the double-bond and the charge-separated forms, because the dataset SMILES use both.
FUNCTIONAL_GROUPS_V1: tuple[tuple[str, str], ...] = (
    # Carbonyl and carboxyl derivatives.
    ("carboxylic_acid", "[CX3](=O)[OX2H1]"),
    ("carboxylic_ester", "[#6][CX3](=O)[OX2H0][#6]"),
    ("lactone", "[#6][CX3;R](=O)[OX2H0;R][#6]"),
    ("anhydride", "[CX3](=O)[OX2][CX3](=O)"),
    ("acyl_halide", "[CX3](=O)[F,Cl,Br,I]"),
    ("aldehyde", "[CX3H1](=O)[#6]"),
    ("ketone", "[#6][CX3](=O)[#6]"),
    ("quinone", "[#6]1(=O)[#6]=,:[#6][#6](=O)[#6]=,:[#6]1"),
    ("primary_amide", "[CX3](=O)[NX3H2]"),
    ("substituted_amide", "[#6][CX3](=O)[NX3;H0,H1][#6]"),
    ("lactam", "[#6][CX3;R](=O)[NX3;R]"),
    ("beta_lactam", "[CX3;r4](=O)[NX3;r4]"),
    ("imide", "[CX3](=O)[NX3][CX3](=O)"),
    ("urea", "[NX3][CX3](=O)[NX3]"),
    ("carbamate", "[NX3][CX3](=O)[OX2][#6]"),
    # Nitrogen groups.
    ("guanidine", "[NX3][CX3](=[NX2])[NX3]"),
    ("amidine", "[#6][CX3](=[NX2])[NX3]"),
    ("imine", "[CX3]=[NX2;!$(N-[OX2])][#6]"),
    ("oxime", "[CX3]=[NX2][OX2]"),
    ("hydrazine_or_hydrazone", "[N;!a]-[N;!a]"),
    ("azo", "[#6][NX2]=[NX2][#6]"),
    ("nitro", "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
    ("nitroso", "[#6][NX2]=O"),
    ("nitrile", "[NX1]#[CX2]"),
    ("iso_thio_cyanate", "[NX2]=[CX2]=[OX1,SX1]"),
    ("azide", "[NX2]=[NX2+]=[NX1-]"),
    ("primary_aliphatic_amine", "[NX3;H2;!$(N[#6]=[O,S,N]);!$(N-a)][CX4]"),
    ("secondary_aliphatic_amine", "[NX3;H1;!$(N[#6]=[O,S,N]);!$(N-a)]([CX4])[CX4]"),
    ("tertiary_aliphatic_amine", "[NX3;H0;!$(N[#6]=[O,S,N]);!$(N-a)]([CX4])([CX4])[CX4]"),
    ("aromatic_amine", "[NX3;!$(N[#6]=[O,S,N]);!$(N=*);!$(N-[#7,#8,#16])]-a"),
    ("quaternary_ammonium", "[NX4+]"),
    ("n_oxide", "[#7+;!$([#7+]=O)][OX1-]"),
    # Aromatic heterocycles.
    ("pyridine_like_n", "[nX2;r6]"),
    ("pyrrole_like_nh", "[nX3;H1]"),
    ("imidazole", "[#7;a;r5]1:[#6;a]:[#7;a]:[#6;a]:[#6;a]:1"),
    ("indole", "[#6]1:[#6]:[#6]:[#6]2:[#7]:[#6]:[#6]:[#6]:2:[#6]:1"),
    ("purine", "[#6]1:[#7]:[#6]:[#6]2:[#7]:[#6]:[#7]:[#6]:2:[#7]:1"),
    ("coumarin", "O=c1ccc2ccccc2o1"),
    ("furan_like_o", "[oX2]"),
    ("thiophene_like_s", "[sX2]"),
    # Oxygen groups.
    ("aliphatic_alcohol", "[OX2H][CX4]"),
    ("phenol", "[OX2H]c"),
    ("dialkyl_ether", "[OX2]([CX4])[CX4]"),
    ("aryl_ether", "[OX2](c)[#6;!$([#6]=O)]"),
    ("acetal_like", "[CX4;!$(C(O)(O)O)]([OX2])[OX2]"),
    ("epoxide", "[#6]1[OX2][#6]1"),
    ("peroxide", "[OX2][OX2]"),
    # Sulfur groups.
    ("thiol", "[SX2H]"),
    ("thioether", "[#6][SX2;!a][#6]"),
    ("disulfide", "[SX2][SX2]"),
    ("thiocarbonyl", "[#6,#7][CX3]=[SX1]"),
    ("sulfoxide", "[#6][$([SX3]=O),$([SX3+][OX1-])][#6]"),
    ("sulfone", "[#6][SX4](=O)(=O)[#6]"),
    ("sulfonamide", "[SX4](=O)(=O)[NX3]"),
    ("sulfonic_acid_or_sulfate", "[SX4](=O)(=O)[OX2,OX1-]"),
    # Phosphorus groups.
    ("phosphoryl", "[$([PX4]=O),$([PX4+][OX1-])]"),
    ("thiophosphoryl", "[$([PX4]=S),$([PX4+][SX1-])]"),
    ("trivalent_phosphorus", "[PX3]"),
    # Halogens.
    ("aryl_fluoride", "F[c]"),
    ("aryl_chloride", "Cl[c]"),
    ("aryl_bromide", "Br[c]"),
    ("aryl_iodide", "I[c]"),
    ("trifluoromethyl", "[CX4](F)(F)F"),
    ("alkyl_fluoride", "F[CX4]"),
    ("alkyl_chloride", "Cl[CX4]"),
    ("alkyl_bromide", "Br[CX4]"),
    ("alkyl_iodide", "I[CX4]"),
    # Unusual elements.
    ("boron", "[#5]"),
    ("silicon", "[#14]"),
    ("arsenic", "[#33]"),
    ("selenium", "[#34]"),
    (
        "other_element",
        "[!#1;!#5;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#33;!#34;!#35;!#53]",
    ),
    # Unsaturation and ring systems.
    ("alkyne", "[CX2]#[CX2]"),
    ("macrocycle", "[r{12-}]"),
    (
        "steroid_core",
        "[#6]1~[#6]~[#6]~[#6]2~[#6](~[#6]~1)~[#6]~[#6]~[#6]1~[#6]~2~[#6]~[#6]~[#6]2~[#6]~[#6]"
        "~[#6]~[#6]~1~2",
    ),
)

CLASS_PANELS: dict[str, tuple[tuple[str, str], ...]] = {
    "functional_groups_v1": FUNCTIONAL_GROUPS_V1,
}
WEIGHTING_MODES = frozenset({"uniform", "class"})
# Raw CE columns that define the recorded CE signature.
RAW_CE_COLS = ("ce_type", "nce", "nce_extra_1", "nce_extra_2", "ace", "ace_extra_1", "ace_extra_2")


def molecule_keys(spec_df: pd.DataFrame, mol_df: pd.DataFrame) -> np.ndarray:
    """Map each spectrum to its 2D molecule identity.

    Args:
        spec_df: Spectrum rows with a ``mol_id`` column, shape ``(num_spectra, ...)``.
        mol_df: Molecule table indexed by ``mol_id`` with an ``inchikey_s`` column.

    Returns:
        Object array of ``inchikey_s`` strings, shape ``(num_spectra,)``.

    Raises:
        ValueError: If any spectrum's molecule has no 2D InChIKey block.
    """
    keys = mol_df.loc[spec_df["mol_id"].to_numpy(), "inchikey_s"].to_numpy(dtype=object)
    missing = pd.isna(keys) | (keys == "")
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} spectra have no inchikey_s; molecule identity is undefined"
        )
    return keys


def _ce_token(value: object) -> str:
    """Format one raw CE field, keeping missingness explicit."""
    if pd.isna(value) or value == -1:
        return "NA"
    return repr(float(value))


def acquisition_keys(spec_df: pd.DataFrame, raw_ce_df: pd.DataFrame) -> np.ndarray:
    """Build the per-spectrum acquisition key from recorded metadata.

    The key is ``adduct|instrument|frag_mode|ce_type|nce,nce1,nce2|ace,ace1,ace2``. NCE and
    ACE stay separate, missing values are the literal ``NA``, and stepped or ramped schedules
    keep every recorded value. The molecule is not part of this string; callers pair it with
    the molecule key.

    Args:
        spec_df: Dataset spectrum rows with ``spec_id``, ``prec_type``, ``inst_type`` and
            ``frag_mode``, shape ``(num_spectra, ...)``.
        raw_ce_df: Raw CE columns ``RAW_CE_COLS`` indexed by ``spec_id``, before imputation.

    Returns:
        Object array of key strings, shape ``(num_spectra,)``.

    Raises:
        ValueError: If a spectrum is absent from ``raw_ce_df``.
    """
    missing_ids = ~spec_df["spec_id"].isin(raw_ce_df.index)
    if missing_ids.any():
        raise ValueError(f"{int(missing_ids.sum())} spectra have no raw CE row")
    raw = raw_ce_df.loc[spec_df["spec_id"].to_numpy()]
    keys = []
    for meta, ce in zip(
        spec_df[["prec_type", "inst_type", "frag_mode"]].itertuples(index=False, name=None),
        raw[list(RAW_CE_COLS)].itertuples(index=False, name=None),
    ):
        ce_type = "NA" if pd.isna(ce[0]) else str(ce[0])
        nce = ",".join(_ce_token(v) for v in ce[1:4])
        ace = ",".join(_ce_token(v) for v in ce[4:7])
        meta_str = "|".join("NA" if pd.isna(v) else str(v) for v in meta)
        keys.append(f"{meta_str}|{ce_type}|nce={nce}|ace={ace}")
    return np.asarray(keys, dtype=object)


def match_class_panel(smiles_by_mol: pd.Series, panel_name: str) -> pd.DataFrame:
    """Match every molecule against a frozen functional-group panel.

    Args:
        smiles_by_mol: SMILES indexed by molecule key, shape ``(num_molecules,)``.
        panel_name: Key of ``CLASS_PANELS``.

    Returns:
        Boolean matrix indexed by molecule key with one column per class,
        shape ``(num_molecules, num_classes)``.

    Raises:
        ValueError: If the panel is unknown, a SMARTS fails to parse, or a SMILES fails.
    """
    if panel_name not in CLASS_PANELS:
        raise ValueError(
            f"unknown class panel '{panel_name}'; expected one of {sorted(CLASS_PANELS)}"
        )
    (Chem,) = rdkit_import("rdkit.Chem")
    panel = CLASS_PANELS[panel_name]
    patterns = []
    for name, smarts in panel:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            raise ValueError(f"class '{name}' has invalid SMARTS {smarts}")
        patterns.append(pattern)
    matches = np.zeros((len(smiles_by_mol), len(panel)), dtype=bool)
    for row, smiles in enumerate(smiles_by_mol.to_numpy()):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"cannot parse SMILES {smiles}")
        for col, pattern in enumerate(patterns):
            matches[row, col] = mol.HasSubstructMatch(pattern)
    return pd.DataFrame(matches, index=smiles_by_mol.index, columns=[name for name, _ in panel])


def class_rarity_weights(class_matrix: pd.DataFrame, ref_count: float, cap: float) -> pd.Series:
    """Weight each molecule by its rarest matched class.

    A class matched by ``n`` distinct training molecules has weight
    ``min(cap, max(1, sqrt(ref_count / n)))``. A molecule takes the maximum weight over its
    classes, never their sum; a molecule with no matched class has weight 1.

    Args:
        class_matrix: Boolean molecule-by-class matrix, shape ``(num_molecules, num_classes)``.
        ref_count: Class size at and above which a class has weight 1.
        cap: Largest allowed weight.

    Returns:
        Weights indexed like ``class_matrix``, shape ``(num_molecules,)``.
    """
    counts = class_matrix.sum(axis=0).to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore"):
        class_weight = np.clip(np.sqrt(ref_count / counts), 1.0, cap)
    weighted = np.where(class_matrix.to_numpy(), class_weight[None, :], 1.0)
    weights = weighted.max(axis=1) if weighted.shape[1] else np.ones(len(class_matrix))
    return pd.Series(weights, index=class_matrix.index)


def mixture_probabilities(weights: np.ndarray, uniform_mix: float) -> np.ndarray:
    """Mix uniform and weight-proportional molecule probabilities.

    ``q(m) = uniform_mix / M + (1 - uniform_mix) * w(m) / sum(w)``.

    Args:
        weights: Positive molecule weights, shape ``(num_molecules,)``.
        uniform_mix: Probability mass assigned uniformly, in ``[0, 1]``.

    Returns:
        Probabilities summing to one, shape ``(num_molecules,)``.
    """
    weights = np.asarray(weights, dtype=np.float64)
    return uniform_mix / len(weights) + (1.0 - uniform_mix) * weights / weights.sum()


def build_molecule_acquisition_sampler(
    ds, config_d: dict, generator: th.Generator
) -> MoleculeAcquisitionSampler:
    """Build the A1/A3 training sampler from a filtered training dataset.

    Args:
        ds: Training dataset; positional row ``i`` of ``ds.spec_df`` is sample ``i``.
        config_d: Experiment config with the ``molecule_acquisition_sampler_*`` keys and
            ``spec_fp``.
        generator: Generator the training loop reseeds every epoch.

    Returns:
        Sampler drawing ``len(ds)`` indices per epoch.

    Raises:
        ValueError: If the dataset is merged or its metadata is incomplete.
    """
    if ds.spec_params["merge"]:
        raise ValueError("molecule_acquisition_sampler requires unmerged spectra")
    spec_df = ds.spec_df

    # 1. Molecule and acquisition identity per spectrum.
    mol_keys = molecule_keys(spec_df, ds.mol_df)
    raw_ce_df = pd.read_pickle(config_d["spec_fp"])
    raw_ce_df = raw_ce_df.loc[raw_ce_df["spec_id"].isin(spec_df["spec_id"])]
    raw_ce_df = raw_ce_df.set_index("spec_id")[list(RAW_CE_COLS)]
    acq_keys = acquisition_keys(spec_df, raw_ce_df)
    mol_codes, mol_uniques = pd.factorize(mol_keys, sort=True)
    acq_codes, acq_uniques = pd.factorize(
        pd.Series(mol_keys, dtype=object) + "||" + pd.Series(acq_keys, dtype=object), sort=True
    )

    # 2. Molecule weights from training counts only.
    weighting = config_d["molecule_acquisition_sampler_weighting"]
    if weighting == "uniform":
        weights = np.ones(len(mol_uniques))
    else:
        first_mol_id = (
            pd.DataFrame({"key": mol_keys, "mol_id": spec_df["mol_id"].to_numpy()})
            .groupby("key", sort=True)["mol_id"]
            .first()
        )
        smiles = pd.Series(
            ds.mol_df.loc[first_mol_id.to_numpy(), "smiles"].to_numpy(), index=first_mol_id.index
        )
        class_matrix = match_class_panel(
            smiles, config_d["molecule_acquisition_sampler_class_panel"]
        )
        weights = (
            class_rarity_weights(
                class_matrix,
                ref_count=config_d["molecule_acquisition_sampler_ref_count"],
                cap=config_d["molecule_acquisition_sampler_weight_cap"],
            )
            .loc[mol_uniques]
            .to_numpy()
        )
        class_counts = class_matrix.sum(axis=0).sort_values()
        logger.info(
            "class panel %s: %d molecules match no class; rarest classes %s",
            config_d["molecule_acquisition_sampler_class_panel"],
            int((~class_matrix.any(axis=1)).sum()),
            class_counts.head(10).to_dict(),
        )
    probs = mixture_probabilities(weights, config_d["molecule_acquisition_sampler_uniform_mix"])

    # 3. Report the intended exposure before training.
    logger.info(
        "molecule_acquisition_sampler[%s]: %d spectra, %d molecules, %d molecule-acquisition "
        "keys; %d molecules weighted above 1; molecule effective sample size %.1f",
        weighting,
        len(spec_df),
        len(mol_uniques),
        len(acq_uniques),
        int((weights > 1.0).sum()),
        1.0 / float(np.square(probs).sum()),
    )
    return MoleculeAcquisitionSampler(
        mol_codes=mol_codes,
        acq_codes=acq_codes,
        mol_probs=probs,
        num_samples=len(ds),
        generator=generator,
    )
