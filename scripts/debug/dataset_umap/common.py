"""Shared pieces of the dataset chemical-space pipeline in ``scripts/debug/dataset_umap``.

Steps (run from the repository root; findings in ``docs/findings/casmi2026_halogen_census_2026_10_16.md``):

1. ``collect_mols.py`` (``FRAGNNET-GPU``: the processed ``mol_df.pkl`` files hold RDKit objects).
   One row per (dataset, 2D InChIKey block) with SMILES, spectrum count and Br/Cl/I flags; optional
   structure-only sets (HMDB, ``--extra``) and the Spectraverse per-molecule library sources.
2. ``plot_umap.py`` (``umap-viz``: umap-learn 0.5.7; ``FRAGNNET-GPU`` has no UMAP). Fits one joint
   UMAP of the MS/MS datasets, draws the per-dataset and HMDB / NP Atlas overlap figures, saves
   coordinates and ``umap_model.pkl``.
3. ``nn_fit.py``: nearest-molecule similarity of one dataset (HMDB, NP Atlas) to each MS/MS dataset.
4. ``project.py``: places such a dataset on a fitted map with ``reducer.transform``.
5. ``open_vs_nist.py``: what open MS/MS data covers of NIST20 ∪ NIST23 HR.
6. ``combo_coverage.py``: coverage of candidate open training-set combinations.

Steps 2-6 run with ``python -I``, which leaves the script directory off ``sys.path``; each script
puts it back before importing this module. Imports here stay light (numpy, pandas, RDKit) so the
module loads in both environments.
"""

from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

# Datasets, in figure order. A dataset keeps its colour (reference categorical palette slots 1-8)
# in every figure.
NIST = ["NIST20 HR", "NIST23 HR"]
MSMS_REFS = [*NIST, "MassSpecGym", "Spectraverse", "MSnLib", "Enveda-180"]
NP_PROXY = "CASMI NP proxy"
ORDER = [*MSMS_REFS, NP_PROXY, "HMDB"]
DATASET_COLORS = dict(
    zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"])
)
GREY = "#e4e3de"
DATASET_GREY = "#9a998f"
INK, MUTED = "#1a1a19", "#6b6a63"

# Morgan fingerprints for the map (Jaccard) and for nearest-molecule Tanimoto.
FP_RADIUS, FP_BITS = 2, 2048

# Nearest-molecule Tanimoto bins (lower, upper, label, colour), most similar first. No measured
# similarity lies in [0.999, 1), so 0.999 means an identical fingerprint.
IDENTICAL = 0.999
SIMILARITY_BINS = [
    (IDENTICAL, 1.01, "identical (same 2D molecule or fingerprint)", "#2a78d6"),
    (0.7, IDENTICAL, "close analogue ≥0.7", "#1baf7a"),
    (0.4, 0.7, "related 0.4–0.7", "#eda100"),
    (0.0, 0.4, "no analogue <0.4", "#e34948"),
]


def with_spectra(mols: pd.DataFrame) -> pd.DataFrame:
    """The MS/MS datasets of a ``collect_mols.py`` table: drops structure-only sets (HMDB, ``--extra``).

    They have no spectra, stay out of the map fit (a joint fit with HMDB fails spectral
    initialisation) and never count as open MS/MS data; ``project.py`` places them instead.
    """
    spectra = mols.groupby("dataset")["n_spectra"].sum()
    return mols[mols["dataset"].isin(spectra[spectra > 0].index)]


def slug(name: str) -> str:
    """File-name stem of a dataset name ("NP Atlas" -> "npatlas")."""
    return name.lower().replace(" ", "")


def mol_info(smiles: str) -> tuple[str, str | None, bool, bool, bool]:
    """(smiles, 2D InChIKey block or None, Br, Cl, I carrier) from RDKit, the same key for every source."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, None, False, False, False
    symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    return smiles, Chem.MolToInchiKey(mol)[:14] or None, "Br" in symbols, "Cl" in symbols, "I" in symbols


def mol_info_frame(smiles: list[str], workers: int) -> pd.DataFrame:
    """``mol_info`` for unique SMILES, in parallel; columns smiles, ik14, Br, Cl, I."""
    with Pool(workers) as pool:
        info = pool.map(mol_info, list(dict.fromkeys(smiles)), chunksize=2000)
    return pd.DataFrame(info, columns=["smiles", "ik14", "Br", "Cl", "I"])


def morgan(smiles: str):
    """Morgan bit vector, or None when RDKit cannot parse the SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    return gen.GetFingerprint(mol) if mol is not None else None


def fingerprints(smiles: list[str], n_bits: int = FP_BITS) -> np.ndarray:
    """Boolean fingerprint matrix for UMAP; an unparsable SMILES gives an all-zero row."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=bool)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr := np.zeros(n_bits, dtype=np.uint8))
            out[i] = arr.astype(bool)
    return out


_REF: list = []


def _init(ref_smiles: list[str]) -> None:
    global _REF
    _REF = [fp for fp in map(morgan, ref_smiles) if fp is not None]


def _nn(smiles_chunk: list[str]) -> list[float]:
    out = []
    for smi in smiles_chunk:
        fp = morgan(smi)
        out.append(max(DataStructs.BulkTanimotoSimilarity(fp, _REF)) if fp is not None else np.nan)
    return out


def nearest(query: list[str], ref: list[str], workers: int) -> np.ndarray:
    """Per query SMILES, the highest Tanimoto to any reference SMILES (NaN if unparsable)."""
    chunks = [query[i : i + 200] for i in range(0, len(query), 200)]
    with Pool(workers, initializer=_init, initargs=(ref,)) as pool:
        return np.array([v for part in pool.map(_nn, chunks) for v in part])


def similarity_bin(nn) -> pd.Series:
    """The ``SIMILARITY_BINS`` label of each nearest-molecule similarity."""
    edges = [b[0] for b in SIMILARITY_BINS[::-1]] + [SIMILARITY_BINS[0][1]]
    return pd.cut(pd.Series(nn), edges, right=False, labels=[b[2] for b in SIMILARITY_BINS[::-1]])


def similarity_summary(nn) -> dict:
    """Median and the identical / ≥0.7 / <0.4 fractions of nearest-molecule similarities."""
    nn = pd.Series(nn)
    return {"median": round(float(nn.median()), 3), "frac_ge_0.7": round(float((nn >= 0.7).mean()), 3),
            "frac_lt_0.4": round(float((nn < 0.4).mean()), 3),
            "frac_identical": round(float((nn >= IDENTICAL).mean()), 3)}


def view_limits(xy: pd.DataFrame, pct: float) -> list[np.ndarray]:
    """x and y limits from the pct..100-pct percentiles, padded by 8%.

    Small disconnected clusters stretch the full range and squeeze the main cloud; callers report
    how many points fall outside the view.
    """
    lims = [np.percentile(xy[c], [pct, 100 - pct]) for c in ("x", "y")]
    return [lo_hi + np.array([-0.08, 0.08]) * (lo_hi[1] - lo_hi[0]) for lo_hi in lims]


def in_view(xy: pd.DataFrame, lims: list[np.ndarray]) -> pd.Series:
    return xy["x"].between(*lims[0]) & xy["y"].between(*lims[1])


def base_map(ax, coords: pd.DataFrame, lims: list[np.ndarray]) -> None:
    """Zoom to ``lims`` and draw every mapped molecule in light grey underneath."""
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    ax.scatter(coords["x"], coords["y"], s=0.2, c=GREY, linewidths=0, rasterized=True)


def style(ax, title: str, subtitle: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="semibold", pad=14)
    ax.text(0, 1.005, subtitle, transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")


def note(ax, text: str) -> None:
    ax.text(0.99, 0.01, text, transform=ax.transAxes, fontsize=8, color=MUTED, ha="right", va="bottom")
