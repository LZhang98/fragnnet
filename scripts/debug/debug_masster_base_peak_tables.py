"""Group a MASSter checkpoint's base-peak errors by chemistry, with matched comparisons.

Stage 2 of the wrong-peak analysis (2026-09-28), CPU only, from the tables written by
``debug_masster_base_peak_errors.py``. Each candidate structure (fragment atoms plus H shift)
is annotated with:

- its neutral loss formula and a named loss class;
- its cut bonds, each given a functional class (amide C-N, ester acyl-O, benzylic C-C, ...);
- whether it keeps a whole match of the molecule's most basic site class (molecules without
  a basic N or P site have none), whether it keeps, loses whole or breaks aromatic rings, and
  how much of the largest conjugated system it keeps.

A peak several structures can explain keeps all of them. Its loss is shared (same m/z, same
formula); for structure-level attributes the peak takes the prediction-weighted mean over its
structures, and it is flagged ambiguous when structures holding at least 5% of its predicted
share disagree.

Tables:

1. Base-peak outcome and the cosine split, per split.
2. Hit rate (covered spectra only) by instrument, NCE, novelty and the measured base peak's
   chemistry, train against val.
3. Anatomy of the wrong choices: the model's top peak against the measured base peak.
4. Matched pairwise test. For the measured #1 and #2 peaks of each spectrum, the model is
   right when it predicts #1 above #2. For a binary attribute X that differs between them,
   ``bias = acc(X on #1) - acc(X on #2)``: positive when the model favors X-candidates
   beyond what the targets support, negative when it disfavors them. The same chemistry on
   both sides of the comparison keeps common chemistry from looking like an error. 95%
   intervals come from a molecule-clustered bootstrap.
5. The same comparison as one joint regression of the log-ratio residual, so correlated
   attributes are read holding the others fixed.
6. Calibration in the large: per spectrum, the predicted against the target mean of each
   attribute over all candidates. Tables 4 and 5 select pairs by target rank, so a model that
   cannot tell two candidates apart and falls back on the population prior shows a bias there
   even when it is calibrated in aggregate. This table has no such selection.
7. Cross-fitted reweighting: ``p' ∝ p^γ exp(β·x)`` over the saved candidates, fitted to
   maximize mean cosine on one half of the molecules and scored on the other. ``γ`` alone is
   the calibrated-cosine baseline; the gain from adding ``β`` is what correcting the
   attribute biases in aggregate is worth.
8. Pairwise accuracy by how the measured #1 and #2 relate: nested (one step further along a
   path), overlapping siblings or disjoint (which bond breaks), complementary (which side of
   one cleavage keeps the charge), or the same atoms with different H. Nested pairs are then
   split by which one the target favors, by NCE, novelty and the step between them.

Run from the repository root:

    python scripts/debug/debug_masster_base_peak_tables.py \
        --inputs output/base_peak_errors/i1_e059_val.pkl output/base_peak_errors/i1_e059_train4k.pkl \
        --meta output/i1_memorization_2026_09_28/meta.pkl --output_dp output/base_peak_errors
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.optimize import minimize

RDLogger.DisableLog("rdApp.*")
LOGGER = logging.getLogger(__name__)

# Basic-site classes, most basic first (as in the i1 residual-structure probe, without its
# carbonyl fallback: a molecule with none of these has no basic site here).
BASE_SMARTS = [
    ("quaternary", "[N+,P+;!$([N+][O-]);!$([N+]=O);H0]"),
    ("guanidine", "[NX3][CX3](=[NX2])[NX3]"),
    ("aliphatic_amine", "[NX3;!$(N[#6]=[#7,#8,#16]);!$(N[a]);!$(N[S,P]=O);!$(N-[N,O]);!$(N#*)]"),
    ("amidine_or_imine", "[CX3](=[NX2;!$(N-[O,N])])[#6,#7,#1]"),
    ("pyridine_like", "[nX2;r6]"),
    ("imidazole_like", "[nX2;r5]"),
    ("aniline", "[NX3;H2,H1,H0;$(N[a]);!$(N[#6]=[#7,#8,#16]);!$(N[S,P]=O)]"),
]

# Cut-bond classes, first match wins. Atom maps 1 and 2 mark the bond's endpoints.
BOND_SMARTS = [
    ("amide_CN", "[CX3:1](=O)-[#7:2]"),
    ("ester_acylO", "[CX3:1](=O)-[OX2:2][#6]"),
    ("acid_CO", "[CX3:1](=O)-[OX2H1:2]"),
    ("sulfonyl_SN", "[SX4:1](=O)(=O)-[#7:2]"),
    ("sulfonyl_SC", "[SX4:1](=O)(=O)-[#6:2]"),
    ("phosph_PO", "[P:1]-[O:2]"),
    ("anomeric_CO", "[CX4;R:1]([OX2;R])-[OX2;!R:2]"),
    ("aryl_O", "[c:1]-[OX2:2]"),
    ("alkyl_O", "[CX4:1]-[OX2:2]"),
    ("aryl_N", "[c:1]-[#7;!a:2]"),
    ("alkyl_N", "[CX4:1]-[#7:2]"),
    ("C_halogen", "[#6:1]-[F,Cl,Br,I:2]"),
    ("C_S", "[#6:1]-[#16:2]"),
    ("N_N", "[#7:1]-[#7:2]"),
    ("N_O", "[#7:1]-[#8:2]"),
    ("acyl_CC", "[CX3:1](=O)-[#6:2]"),
    ("biaryl_CC", "[c:1]-!@[c:2]"),
    ("benzylic_CC", "[c:1]-[CX4:2]"),
    ("alkyl_CC", "[CX4:1]-[CX4:2]"),
]

# Named neutral losses (Hill formulas). Anything else is binned by loss mass.
NAMED_LOSSES = {
    "": "none",
    "H2O": "H2O",
    "H3N": "NH3",
    "CO": "CO",
    "CO2": "CO2",
    "CH2O2": "HCOOH",
    "CH4O": "CH3OH",
    "C2H4": "C2H4",
    "C2H2O": "ketene",
    "ClH": "HCl",
    "FH": "HF",
    "H2S": "H2S",
    "CHN": "HCN",
    "H4O2": "2H2O",
    "C2H4O2": "AcOH",
    "C6H10O5": "hexose",
    "C3H9N": "trimethylamine",
    "C2H7N": "dimethylamine",
    "CH5N": "methylamine",
    "BrH": "HBr",
    "O2S": "SO2",
    "H": "H",
    "H2": "H2",
    "CH3": "CH3",
}

ELEMENT_MASS = {
    "C": 12.0,
    "H": 1.00782503207,
    "N": 14.0030740048,
    "O": 15.99491461956,
    "S": 31.97207100,
    "P": 30.97376163,
    "F": 18.99840322,
    "Cl": 34.96885268,
    "Br": 78.9183371,
    "I": 126.904473,
    "Si": 27.9769265325,
    "B": 11.0093054,
    "Se": 79.9165213,
    "As": 74.9215965,
    "Na": 22.9897692809,
    "K": 38.96370668,
}

NCE_BINS = [0, 20, 30, 40, 60, 1e9]
NCE_LABELS = ["<=20", "20-30", "30-40", "40-60", ">60"]
NOVELTY_BINS = [-1, 0.4, 0.6, 0.8, 2.0]
NOVELTY_LABELS = ["<0.4", "0.4-0.6", "0.6-0.8", ">=0.8"]


def hill(counts: Counter) -> str:
    """Hill-order formula string; zero and negative counts are dropped."""
    items = {k: v for k, v in counts.items() if v > 0}
    keys = (["C", "H"] if "C" in items else []) + sorted(
        k for k in items if k not in (("C", "H") if "C" in items else ())
    )
    return "".join(f"{k}{items[k] if items[k] > 1 else ''}" for k in keys if k in items)


def parse_formula(formula: str) -> Counter:
    """Inverse of ``hill``: element counts of a formula string."""
    return Counter({e: int(n or 1) for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula)})


def formula_difference(deeper: str, shallower: str) -> str:
    """What one candidate loses beyond another, from their loss formulas.

    Returns the named loss (``H2O``, ``CO``, ...) or its mass bin when every element count of
    ``deeper - shallower`` is non-negative, ``same`` when they are equal and ``mixed``
    otherwise (neither ion's formula contains the other's).
    """
    delta = parse_formula(deeper)
    delta.subtract(parse_formula(shallower))
    if any(v < 0 for v in delta.values()):
        return "mixed"
    formula = hill(delta)
    return "same" if not formula else loss_class(formula)


def loss_class(formula: str) -> str:
    """Named loss, or a mass bin for anything unnamed."""
    if formula in NAMED_LOSSES:
        return NAMED_LOSSES[formula]
    mass = sum(ELEMENT_MASS[e] * int(n or 1) for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula))
    for upper in (50, 100, 150, 250):
        if mass < upper:
            return f"other<{upper}"
    return "other>=250"


class MoleculeChemistry:
    """Per-molecule lookups for annotating candidate structures (atom indices are RDKit's)."""

    def __init__(self, mol: Chem.Mol):
        self.mol = mol
        self.n_atoms = mol.GetNumAtoms()
        self.symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        self.hydrogens = [a.GetTotalNumHs() for a in mol.GetAtoms()]
        self.bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
        self.bond_ring = [b.IsInRing() for b in mol.GetBonds()]
        self.bond_class = self._bond_classes()
        self.formula = Counter()
        for symbol, hydrogens in zip(self.symbols, self.hydrogens, strict=True):
            self.formula[symbol] += 1
            self.formula["H"] += hydrogens
        # Most basic site class present, and each of its matches.
        self.base_class, self.base_matches = "none", ()
        for name, smarts in BASE_SMARTS:
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
            if matches:
                self.base_class = name
                self.base_matches = tuple(frozenset(m) for m in matches)
                break
        ring_info = mol.GetRingInfo()
        self.aromatic_rings = [
            frozenset(r)
            for r in ring_info.AtomRings()
            if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r)
        ]
        self.conjugated = self._largest_conjugated_system()

    def _bond_classes(self) -> list[str]:
        classes = [None] * len(self.bonds)
        index = {frozenset(b): i for i, b in enumerate(self.bonds)}
        for name, smarts in BOND_SMARTS:
            pattern = Chem.MolFromSmarts(smarts)
            mapped = {
                a.GetAtomMapNum(): a.GetIdx() for a in pattern.GetAtoms() if a.GetAtomMapNum()
            }
            for match in self.mol.GetSubstructMatches(pattern, uniquify=False):
                i = index.get(frozenset((match[mapped[1]], match[mapped[2]])))
                if i is not None and classes[i] is None:
                    classes[i] = name
        for i, (u, v) in enumerate(self.bonds):
            if classes[i] is None:
                a, b = self.mol.GetAtomWithIdx(u), self.mol.GetAtomWithIdx(v)
                ends = sorted(f"{x.GetSymbol()}{'ar' if x.GetIsAromatic() else ''}" for x in (a, b))
                classes[i] = "other_" + "-".join(ends)
            if self.bond_ring[i]:
                classes[i] = "ring:" + classes[i]
        return classes

    def _largest_conjugated_system(self) -> frozenset:
        parent = list(range(self.n_atoms))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for bond in self.mol.GetBonds():
            if bond.GetIsConjugated():
                parent[find(bond.GetBeginAtomIdx())] = find(bond.GetEndAtomIdx())
        members = Counter(
            find(b.GetBeginAtomIdx()) for b in self.mol.GetBonds() if b.GetIsConjugated()
        )
        if not members:
            return frozenset()
        root = members.most_common(1)[0][0]
        return frozenset(a for a in range(self.n_atoms) if find(a) == root)

    def annotate(self, atoms: tuple[int, ...], h_shift: int) -> dict:
        """Chemistry of one candidate structure (fragment atoms, H shift)."""
        inside = set(atoms)
        fragment = Counter()
        for a in atoms:
            fragment[self.symbols[a]] += 1
            fragment["H"] += self.hydrogens[a]
        fragment["H"] += h_shift
        loss = Counter(self.formula)
        loss.subtract(fragment)
        if any(v < 0 for v in loss.values()):
            raise ValueError(f"negative loss counts for atoms {atoms}, h {h_shift}")
        cuts = [i for i, (u, v) in enumerate(self.bonds) if (u in inside) != (v in inside)]
        loss_formula = hill(loss)
        return {
            "n_atoms": len(atoms),
            "frac_atoms": len(atoms) / self.n_atoms,
            "h_shift": h_shift,
            "is_precursor": len(atoms) == self.n_atoms and h_shift == 0,
            "loss_formula": loss_formula,
            "loss_class": loss_class(loss_formula),
            "n_cuts": len(cuts),
            "cut_classes": tuple(sorted(self.bond_class[i] for i in cuts)),
            "ring_cut": any(self.bond_ring[i] for i in cuts),
            "has_base": bool(self.base_matches),
            "keeps_base": any(m.issubset(inside) for m in self.base_matches),
            "keeps_aromatic": all(r.issubset(inside) for r in self.aromatic_rings),
            "loses_aromatic": any(not (r & inside) for r in self.aromatic_rings),
            "breaks_aromatic": any(0 < len(r & inside) < len(r) for r in self.aromatic_rings),
            "aromatic_rings_kept": sum(r.issubset(inside) for r in self.aromatic_rings),
            "conj_kept": (
                len(self.conjugated & inside) / len(self.conjugated) if self.conjugated else 1.0
            ),
        }


def peak_attributes(structures: list[tuple], chem: MoleculeChemistry, cache: dict) -> dict:
    """Prediction-weighted peak attributes over its structures, with ambiguity flags.

    Args:
        structures: ``(atoms, h_shift, ion_mode, probability)`` tuples at one m/z.
        chem: The molecule's lookups.
        cache: ``(atoms, h_shift) -> annotation`` memo shared across this molecule's spectra.

    Returns:
        Weighted numeric attributes, the loss (shared by construction), the dominant cut
        signature and ``ambiguous_cut`` / ``ambiguous_base`` flags.
    """
    notes, weights = [], []
    for atoms, h_shift, _mode, prob in structures:
        key = (atoms, h_shift)
        if key not in cache:
            cache[key] = chem.annotate(atoms, h_shift)
        notes.append(cache[key])
        weights.append(prob)
    w = np.asarray(weights, float)
    w = w / w.sum() if w.sum() > 0 else np.full(w.size, 1.0 / w.size)
    major = w >= 0.05
    signatures = Counter()
    for note, weight in zip(notes, w, strict=True):
        signatures[note["cut_classes"]] += weight
    out = {
        "loss_formula": notes[0]["loss_formula"],
        "loss_class": notes[0]["loss_class"],
        "n_atoms": notes[0]["n_atoms"],
        "frac_atoms": notes[0]["frac_atoms"],
        "n_structures": len(notes),
        "cut_signature": signatures.most_common(1)[0][0],
        "ambiguous_cut": len({n["cut_classes"] for n, m in zip(notes, major, strict=True) if m})
        > 1,
        "ambiguous_base": len({n["keeps_base"] for n, m in zip(notes, major, strict=True) if m})
        > 1,
        "atoms": structures[int(np.argmax(w))][0],
        "mol_atoms": chem.n_atoms,
    }
    for name in (
        "is_precursor",
        "n_cuts",
        "ring_cut",
        "has_base",
        "keeps_base",
        "keeps_aromatic",
        "loses_aromatic",
        "breaks_aromatic",
        "aromatic_rings_kept",
        "conj_kept",
        "h_shift",
    ):
        out[name] = float(sum(weight * float(n[name]) for n, weight in zip(notes, w, strict=True)))
    out["h_shift_abs"] = float(
        sum(weight * abs(n["h_shift"]) for n, weight in zip(notes, w, strict=True))
    )
    # Weighted share of structures cutting each bond class, for the class-level tests.
    classes = Counter()
    for note, weight in zip(notes, w, strict=True):
        for name in {c.removeprefix("ring:") for c in note["cut_classes"]}:
            classes[name] += weight
    out["cut_class_weights"] = dict(classes)
    return out


def pair_relation(first: tuple, second: tuple, n_atoms: int) -> str:
    """How two candidates' atom sets relate.

    Returns ``same_atoms`` (they differ only in H), ``complementary`` (disjoint and together
    the whole molecule: the two sides of one cleavage), ``disjoint``, ``first_inside_second``,
    ``second_inside_first`` (nested: one lies further along the same path) or ``overlapping``.
    """
    a, b = set(first), set(second)
    if a == b:
        return "same_atoms"
    if not a & b:
        return "complementary" if len(a) + len(b) == n_atoms else "disjoint"
    if a < b:
        return "first_inside_second"
    if b < a:
        return "second_inside_first"
    return "overlapping"


def clustered_bootstrap(
    groups: np.ndarray, values: dict[str, np.ndarray], reps: int, seed: int
) -> np.ndarray:
    """Resample molecules with replacement; return per-rep sums of each value column.

    Args:
        groups: Molecule id per row, shape ``(N,)``.
        values: Column name -> per-row values, each shape ``(N,)``.
        reps: Bootstrap replicates.
        seed: RNG seed.

    Returns:
        Array ``(reps, len(values))`` of resampled column sums, columns in ``values`` order.
    """
    codes, inverse = np.unique(groups, return_inverse=True)
    per_mol = np.stack(
        [np.bincount(inverse, weights=v, minlength=codes.size) for v in values.values()], axis=1
    )
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(codes.size, np.full(codes.size, 1.0 / codes.size), size=reps)
    return counts @ per_mol


def pairwise_bias(
    frame: pd.DataFrame, attribute_first: np.ndarray, attribute_second: np.ndarray, reps: int
) -> dict:
    """Accuracy when the attribute sits on the measured #1 against when it sits on #2.

    Args:
        frame: One row per spectrum pair with ``correct`` (model orders #1 over #2) and
            ``mol_id``.
        attribute_first: Boolean attribute of #1 per row.
        attribute_second: Boolean attribute of #2 per row.
        reps: Bootstrap replicates for the interval on ``bias``.

    Returns:
        Counts, both accuracies, ``bias`` and its 95% molecule-clustered interval.
    """
    on_first = attribute_first & ~attribute_second
    on_second = attribute_second & ~attribute_first
    correct = frame.correct.to_numpy().astype(float)
    n1, n2 = int(on_first.sum()), int(on_second.sum())
    acc1 = correct[on_first].mean() if n1 else np.nan
    acc2 = correct[on_second].mean() if n2 else np.nan
    lo = hi = np.nan
    if n1 >= 20 and n2 >= 20:
        sums = clustered_bootstrap(
            frame.mol_id.to_numpy(),
            {
                "c1": correct * on_first,
                "n1": on_first.astype(float),
                "c2": correct * on_second,
                "n2": on_second.astype(float),
            },
            reps,
            seed=0,
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = sums[:, 0] / sums[:, 1] - sums[:, 2] / sums[:, 3]
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {
        "n_on_first": n1,
        "n_on_second": n2,
        "acc_on_first": acc1,
        "acc_on_second": acc2,
        "bias": acc1 - acc2,
        "lo": lo,
        "hi": hi,
    }


# Structural summary terms of the joint model (Table 5).
JOINT_TERMS = [
    "is_precursor",
    "n_cuts",
    "frac_atoms",
    "ring_cut",
    "h_shift_abs",
    "keeps_base",
    "loses_aromatic",
    "breaks_aromatic",
    "conj_kept",
]


def joint_bias(
    pair: pd.DataFrame, terms: list[str], reps: int, seed: int = 0, floor: float = 1e-6
) -> pd.DataFrame:
    """Regress the pair's log-ratio residual on the attribute differences, jointly.

    The residual is ``log(p1 / p2) - log(t1 / t2)`` for the measured #1 and #2 of a spectrum
    (predictions floored at ``floor``). A positive coefficient on ``attr_1 - attr_2`` means the
    model over-predicts candidates carrying more of the attribute, holding the other terms
    fixed; ``exp(coef)`` is the factor. The intercept absorbs the selection of #1 and #2 by
    target intensity. Intervals: molecule-clustered bootstrap, refitting weighted least squares.

    Args:
        pair: One row per spectrum with ``{term}_1``, ``{term}_2``, ``pred_1``, ``pred_2``,
            ``tgt_1``, ``tgt_2`` and ``mol_id``.
        terms: Attribute names.
        reps: Bootstrap replicates.
        seed: RNG seed.
        floor: Lower bound on predicted shares before the log.

    Returns:
        One row per term (intercept first): ``coef``, ``lo``, ``hi``, and ``n`` pairs.
    """
    x = np.column_stack(
        [np.ones(len(pair))]
        + [pair[f"{t}_1"].to_numpy(float) - pair[f"{t}_2"].to_numpy(float) for t in terms]
    )
    r = (
        np.log(np.maximum(pair.pred_1.to_numpy(float), floor))
        - np.log(np.maximum(pair.pred_2.to_numpy(float), floor))
        - np.log(pair.tgt_1.to_numpy(float) / pair.tgt_2.to_numpy(float))
    )
    coef = np.linalg.lstsq(x, r, rcond=None)[0]
    codes, inverse = np.unique(pair.mol_id.to_numpy(), return_inverse=True)
    rng = np.random.default_rng(seed)
    boot = []
    for counts in rng.multinomial(codes.size, np.full(codes.size, 1.0 / codes.size), size=reps):
        w = counts[inverse].astype(float)
        xw = x * w[:, None]
        boot.append(np.linalg.lstsq(xw.T @ x, xw.T @ r, rcond=None)[0])
    lo, hi = np.percentile(np.asarray(boot), [2.5, 97.5], axis=0)
    return pd.DataFrame(
        {"coef": coef, "lo": lo, "hi": hi, "n": len(pair)}, index=["intercept", *terms]
    )


def reweighted_cosine(
    theta: np.ndarray,
    log_pred: np.ndarray,
    tgt: np.ndarray,
    x: np.ndarray,
    offsets: np.ndarray,
    target_norm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-spectrum cosine of ``p' ∝ exp(γ log p + x β)`` and the gradient of its mean.

    Cosine is scale-invariant, so ``p'`` needs no normalization: with ``e = exp(logit)``,
    ``A = Σ e t`` and ``B = Σ e²`` per spectrum, ``cos = A / (sqrt(B) |T|)`` and
    ``d cos / d logit_i = e_i t_i / (sqrt(B) |T|) - A e_i² / (B^1.5 |T|)``.

    Args:
        theta: ``(γ, β_1..β_A)``.
        log_pred: Log predicted share per saved candidate, shape ``(N,)``.
        tgt: Target intensity per candidate, shape ``(N,)``.
        x: Attributes per candidate, shape ``(N, A)``.
        offsets: Spectrum boundaries into the candidate arrays, shape ``(S + 1,)``; every
            spectrum has at least one candidate.
        target_norm: L2 norm of each whole target, uncovered peaks included, shape ``(S,)``.

    Returns:
        Cosine per spectrum ``(S,)`` and the gradient of its mean with respect to ``theta``.
    """
    starts = offsets[:-1]
    segment = np.repeat(np.arange(starts.size), np.diff(offsets))
    logit = theta[0] * log_pred + x @ theta[1:]
    e = np.exp(logit - np.maximum.reduceat(logit, starts)[segment])
    a = np.add.reduceat(e * tgt, starts)
    b = np.add.reduceat(e * e, starts)
    cos = a / (np.sqrt(b) * target_norm)
    g_logit = e * tgt / (np.sqrt(b) * target_norm)[segment] - (a / (b**1.5 * target_norm))[
        segment
    ] * (e * e)
    grad = np.concatenate([[g_logit @ log_pred], x.T @ g_logit]) / starts.size
    return cos, grad


def fit_reweighting(arrays: dict, rows: np.ndarray, attributes: list[int] | None) -> np.ndarray:
    """Fit ``γ`` (and ``β`` on the attribute columns given) to maximize mean cosine on rows.

    Args:
        arrays: Candidate arrays of one split (``log_pred``, ``tgt``, ``x``, ``offsets``,
            ``target_norm``).
        rows: Spectrum indices to fit on.
        attributes: Attribute columns reweighted, or ``None`` for ``γ`` alone.

    Returns:
        ``theta`` over all attribute columns, zero where not fitted.
    """
    subset = _subset(arrays, rows)
    columns = [] if attributes is None else list(attributes)
    x = subset["x"][:, columns]

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        cos, grad = reweighted_cosine(
            params, subset["log_pred"], subset["tgt"], x, subset["offsets"], subset["target_norm"]
        )
        return -float(cos.mean()), -grad

    start = np.zeros(1 + len(columns))
    start[0] = 1.0
    fitted = minimize(objective, start, jac=True, method="L-BFGS-B").x
    theta = np.zeros(1 + arrays["x"].shape[1])
    theta[0] = fitted[0]
    theta[1 + np.asarray(columns, dtype=int)] = fitted[1:]
    return theta


def _subset(arrays: dict, rows: np.ndarray) -> dict:
    """The candidate arrays of the spectra in ``rows``, re-offset."""
    counts = np.diff(arrays["offsets"])[rows]
    index = np.concatenate(
        [np.arange(arrays["offsets"][r], arrays["offsets"][r + 1]) for r in rows]
    )
    return {
        "log_pred": arrays["log_pred"][index],
        "tgt": arrays["tgt"][index],
        "x": arrays["x"][index],
        "offsets": np.concatenate([[0], np.cumsum(counts)]),
        "target_norm": arrays["target_norm"][rows],
    }


def annotate_split(
    spectra: pd.DataFrame, peaks: pd.DataFrame, molecules: pd.Series
) -> pd.DataFrame:
    """Attach peak attributes to every peak row, one chemistry cache per molecule."""
    peaks = peaks.merge(spectra[["spec_id", "mol_id"]], on="spec_id")
    rows = []
    for mol_id, group in peaks.groupby("mol_id", sort=False):
        chem = MoleculeChemistry(molecules.loc[mol_id])
        cache: dict = {}
        for row in group.itertuples():
            rows.append(
                {
                    "spec_id": row.spec_id,
                    "cand": row.cand,
                    **peak_attributes(row.structures, chem, cache),
                }
            )
    return peaks.drop(columns="structures").merge(pd.DataFrame(rows), on=["spec_id", "cand"])


def main(argv: list[str] | None = None) -> None:
    """Build the four tables for each input split and write them beside the inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="spectrum pickles from stage 1")
    parser.add_argument("--meta", required=True, help="spec_id -> split, inst_type, nce, novelty")
    parser.add_argument("--mol_fp", default="data/proc/nist20/mol_df.pkl")
    parser.add_argument("--output_dp", required=True)
    parser.add_argument("--min_second", type=float, default=0.05, help="#2 / #1 target floor")
    parser.add_argument("--reps", type=int, default=400)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dp = Path(args.output_dp)
    molecules = pd.read_pickle(args.mol_fp).set_index("mol_id")["mol"]
    meta = pd.read_pickle(args.meta).set_index("spec_id")[["split", "inst_type", "nce", "novelty"]]

    # 1. Annotate each split's peaks.
    spectra_all, peaks_all = [], []
    for path in args.inputs:
        spectra = pd.read_pickle(path).join(meta, on="spec_id")
        if spectra.split.isna().any() or spectra.split.nunique() != 1:
            raise ValueError(f"{path}: spectra must map to exactly one split in --meta")
        peaks = pd.read_pickle(Path(path).with_suffix(".peaks.pkl"))
        annotated = annotate_split(spectra, peaks, molecules)
        annotated["split"] = spectra.split.iloc[0]
        spectra_all.append(spectra)
        peaks_all.append(annotated)
        LOGGER.info("%s: annotated %d peaks", path, len(annotated))
    spectra = pd.concat(spectra_all, ignore_index=True)
    peaks = pd.concat(peaks_all, ignore_index=True)
    spectra["nce_bin"] = pd.cut(spectra.nce, NCE_BINS, labels=NCE_LABELS, right=True)
    spectra["novelty_bin"] = pd.cut(spectra.novelty, NOVELTY_BINS, labels=NOVELTY_LABELS)
    peaks.to_pickle(output_dp / "annotated_peaks.pkl")

    # 2. Measured #1 and #2, and the model's top, per spectrum.
    first = peaks[peaks.tgt_rank == 1].set_index("spec_id")
    second = peaks[peaks.tgt_rank == 2].set_index("spec_id")
    model_top = peaks[peaks.pred_rank == 1].set_index("spec_id")
    spectra = spectra.join(first.add_prefix("t1_"), on="spec_id").join(
        model_top.add_prefix("p1_"), on="spec_id"
    )
    spectra["covered"] = spectra.outcome != "uncovered"
    spectra["hit"] = (spectra.outcome == "hit").astype(float)
    spectra.to_pickle(output_dp / "spectra.pkl")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 400)
    out_lines = []

    def emit(title: str, table: pd.DataFrame) -> None:
        text = f"\n## {title}\n{table.to_string()}\n"
        LOGGER.info(text)
        out_lines.append(text)

    # 3. Table 1: overall.
    overall = spectra.groupby("split").agg(
        spectra=("spec_id", "size"),
        molecules=("mol_id", "nunique"),
        cos=("cos", "mean"),
        coverage=("coverage", "mean"),
        ranking=("ranking", "mean"),
        calibration=("calibration", "mean"),
        hit=("outcome", lambda s: (s == "hit").mean()),
        under_ranked=("outcome", lambda s: (s == "under_ranked").mean()),
        uncovered=("outcome", lambda s: (s == "uncovered").mean()),
    )
    emit("Table 1. Outcome and cosine split", overall.round(4))

    # 4. Table 2: hit rate among covered spectra by stratum.
    covered = spectra[spectra.covered].copy()
    covered["t1_kind"] = np.where(covered.t1_is_precursor > 0.5, "precursor", "fragment")
    covered["t1_size_bin"] = pd.cut(covered.t1_frac_atoms, [0, 0.25, 0.5, 0.75, 0.999, 1.0])
    covered["t1_ncuts"] = covered.t1_n_cuts.round().clip(upper=4).astype(int)
    covered["t1_keeps_base"] = np.where(
        covered.t1_has_base > 0.5, np.where(covered.t1_keeps_base > 0.5, "kept", "lost"), "no_base"
    )
    covered["t1_ring_cut"] = covered.t1_ring_cut > 0.5
    for column in (
        "inst_type",
        "nce_bin",
        "novelty_bin",
        "t1_kind",
        "t1_size_bin",
        "t1_ncuts",
        "t1_keeps_base",
        "t1_ring_cut",
        "t1_loss_class",
    ):
        table = (
            covered.groupby([column, "split"], observed=True)
            .agg(n=("hit", "size"), hit=("hit", "mean"))
            .unstack("split")
        )
        table = table[table[("n", "val")] >= 100] if ("n", "val") in table else table
        emit(f"Table 2. Hit rate (covered) by {column}", table.round(3))

    # Measured-base cut class: a spectrum counts under each class its base peak's structures cut.
    rows = []
    for row in covered.itertuples():
        for name, weight in (row.t1_cut_class_weights or {}).items():
            if weight >= 0.5:
                rows.append({"split": row.split, "cls": name, "hit": row.hit})
    if rows:
        cls = (
            pd.DataFrame(rows).groupby(["cls", "split"]).agg(n=("hit", "size"), hit=("hit", "mean"))
        )
        cls = cls.unstack("split")
        cls = cls[cls[("n", "val")] >= 100] if ("n", "val") in cls else cls
        emit("Table 2b. Hit rate (covered) by a cut class of the measured base peak", cls.round(3))

    # 5. Table 3: anatomy of wrong choices (under-ranked spectra).
    wrong = spectra[spectra.outcome == "under_ranked"].copy()
    miss_names = {
        "first_inside_second": "model_top_contains_base_peak",
        "second_inside_first": "base_peak_contains_model_top",
    }
    wrong["relation"] = [
        miss_names.get(r, r)
        for r in map(pair_relation, wrong.t1_atoms, wrong.p1_atoms, wrong.t1_mol_atoms)
    ]
    emit(
        "Table 3a. Model top relative to measured base peak (dominant structures)",
        wrong.groupby(["relation", "split"])
        .size()
        .unstack("split")
        .div(wrong.groupby("split").size())
        .round(3),
    )
    emit(
        "Table 3b. Model top rank of the measured base peak (under-ranked only)",
        wrong.assign(rank=wrong.base_pred_rank.clip(upper=6))
        .groupby(["rank", "split"])
        .size()
        .unstack("split")
        .div(wrong.groupby("split").size())
        .round(3),
    )
    pairs = (
        wrong[wrong.split == "val"]
        .groupby(["t1_loss_class", "p1_loss_class"])
        .size()
        .sort_values(ascending=False)
        .head(30)
        .to_frame("n")
    )
    pairs["share"] = (pairs.n / (wrong.split == "val").sum()).round(3)
    emit("Table 3c. Val wrong choices: measured-base loss -> model-top loss (top 30)", pairs)
    # The formula step between the two: the model's top lost X more (deeper) or X less.
    deeper = [
        formula_difference(p, t)
        for p, t in zip(wrong.p1_loss_formula, wrong.t1_loss_formula, strict=True)
    ]
    shallower = [
        formula_difference(t, p)
        for p, t in zip(wrong.p1_loss_formula, wrong.t1_loss_formula, strict=True)
    ]
    wrong["step"] = [
        (
            f"model_deeper_by_{d}"
            if d != "mixed"
            else (f"model_shallower_by_{h}" if h != "mixed" else "mixed")
        )
        for d, h in zip(deeper, shallower, strict=True)
    ]
    steps = wrong.groupby(["step", "split"]).size().unstack("split").fillna(0)
    steps = steps.div(wrong.groupby("split").size()).sort_values("val", ascending=False).head(25)
    emit(
        "Table 3d. Formula step from measured base peak to model top (under-ranked)", steps.round(3)
    )

    # 6. Table 4: matched pairwise bias between measured #1 and #2.
    pair = first.join(second, lsuffix="_1", rsuffix="_2", how="inner")
    pair = pair[pair.tgt_2 >= args.min_second * pair.tgt_1]
    pair = pair.join(spectra.set_index("spec_id")[["mol_id", "split", "nce_bin", "novelty_bin"]])
    pair["correct"] = pair.pred_1 > pair.pred_2
    attributes = {
        "is_precursor": lambda s: pair[f"is_precursor_{s}"] > 0.5,
        "larger_fragment": lambda s: pair[f"n_atoms_{s}"] > pair[f"n_atoms_{3 - int(s)}"],
        "keeps_base": lambda s: pair[f"keeps_base_{s}"] > 0.5,
        "keeps_all_aromatic": lambda s: pair[f"keeps_aromatic_{s}"] > 0.5,
        "loses_aromatic": lambda s: pair[f"loses_aromatic_{s}"] > 0.5,
        "breaks_aromatic": lambda s: pair[f"breaks_aromatic_{s}"] > 0.5,
        "more_aromatic_rings": lambda s: pair[f"aromatic_rings_kept_{s}"]
        > pair[f"aromatic_rings_kept_{3 - int(s)}"] + 0.5,
        "more_conjugation_kept": lambda s: pair[f"conj_kept_{s}"]
        > pair[f"conj_kept_{3 - int(s)}"] + 0.2,
        "ring_cut": lambda s: pair[f"ring_cut_{s}"] > 0.5,
        "multi_cut": lambda s: pair[f"n_cuts_{s}"] > 1.5,
        "h_transfer": lambda s: pair[f"h_shift_abs_{s}"] > 0.5,
        "more_structures": lambda s: pair[f"n_structures_{s}"] > pair[f"n_structures_{3 - int(s)}"],
    }
    for name in ("H2O", "NH3", "CO", "CO2", "HCOOH", "CH3OH", "HCl", "C2H4", "ketene", "hexose"):
        attributes[f"loss_{name}"] = lambda s, n=name: pair[f"loss_class_{s}"] == n
    class_names = sorted(
        {c for d in pair.cut_class_weights_1.dropna() for c in d}
        | {c for d in pair.cut_class_weights_2.dropna() for c in d}
    )
    for name in class_names:
        attributes[f"cut_{name}"] = lambda s, n=name: pair[f"cut_class_weights_{s}"].map(
            lambda d: (d or {}).get(n, 0.0) >= 0.5
        )
    results = []
    for split in ("val", "train"):
        mask = (pair.split == split).to_numpy()
        if not mask.any():
            continue
        for name, fn in attributes.items():
            a1 = fn("1").to_numpy(bool)[mask]
            a2 = fn("2").to_numpy(bool)[mask]
            stats = pairwise_bias(pair[mask], a1, a2, args.reps)
            results.append({"split": split, "attribute": name, **stats})
    bias = pd.DataFrame(results)
    bias = bias[(bias.n_on_first + bias.n_on_second) >= 100]
    wide = bias.pivot(index="attribute", columns="split")
    emit(
        f"Table 4. Matched pairwise bias, measured #1 vs #2 (#2 >= {args.min_second} x #1); "
        f"pairs: {pair.groupby('split').size().to_dict()}, accuracy "
        f"{pair.groupby('split').correct.mean().round(3).to_dict()}",
        wide[
            [
                c
                for c in wide.columns
                if c[0]
                in (
                    "n_on_first",
                    "n_on_second",
                    "acc_on_first",
                    "acc_on_second",
                    "bias",
                    "lo",
                    "hi",
                )
            ]
        ]
        .round(3)
        .sort_values(("bias", "val")),
    )
    bias.to_csv(output_dp / "pairwise_bias.csv", index=False)

    # 7. Table 5: the joint model, so correlated attributes (depth, aromatic and basic-site
    # retention) are read holding the others fixed.
    for column in class_names:
        for side in ("1", "2"):
            pair[f"cut_{column}_{side}"] = pair[f"cut_class_weights_{side}"].map(
                lambda d, n=column: (d or {}).get(n, 0.0)
            )
    frequent = [
        f"cut_{c}"
        for c in class_names
        if ((pair[f"cut_{c}_1"] - pair[f"cut_{c}_2"]).abs() > 0.5).sum() >= 300
    ]
    unambiguous = ~(pair.ambiguous_cut_1 | pair.ambiguous_cut_2)
    fits = {}
    for split in ("val", "train"):
        in_split = pair.split == split
        if not in_split.any():
            continue
        fits[f"{split}"] = joint_bias(pair[in_split], JOINT_TERMS, args.reps)
        fits[f"{split}_unambiguous"] = joint_bias(
            pair[in_split & unambiguous], JOINT_TERMS, args.reps
        )
        fits[f"{split}_classes"] = joint_bias(pair[in_split], JOINT_TERMS + frequent, args.reps)
        for column in ("nce_bin", "novelty_bin"):
            for level in pair[column].cat.categories:
                subset = pair[in_split & (pair[column] == level)]
                if len(subset) >= 500:
                    fits[f"{split}_{column}={level}"] = joint_bias(subset, JOINT_TERMS, args.reps)
    joint = pd.concat(fits, names=["fit", "term"])
    joint.to_csv(output_dp / "joint_bias.csv")
    summary = joint.coef.unstack("fit")
    emit(
        "Table 5a. Joint log-ratio bias (coef; exp(coef) = fold over-prediction per unit)",
        summary[[f for f in fits if not f.endswith("_classes")]]
        .loc[["intercept", *JOINT_TERMS]]
        .round(3),
    )
    for name in fits:
        if name in ("val", "val_classes", "train", "train_classes", "val_unambiguous"):
            emit(f"Table 5b. Joint fit {name} with 95% interval", fits[name].round(3))
    pair.drop(columns=[c for c in pair.columns if c.startswith("atoms_")]).to_pickle(
        output_dp / "pairs.pkl"
    )

    # 8. Table 6: calibration in the large, over all candidates and no rank selection.
    has_base = {
        int(m): bool(MoleculeChemistry(molecules.loc[m]).base_matches)
        for m in spectra.mol_id.unique()
    }
    spectra["has_base"] = spectra.mol_id.map(has_base)
    calibration_rows = []
    names = [c.removeprefix("ep_") for c in spectra.columns if c.startswith("ep_")]
    for split in spectra.split.unique():
        for stratum, level in [("all", "all")] + [
            (c, v) for c in ("nce_bin", "novelty_bin") for v in spectra[c].cat.categories
        ]:
            chosen = spectra[(spectra.split == split) & spectra.et_n_cuts.notna()]
            if stratum != "all":
                chosen = chosen[chosen[stratum] == level]
            for name in names:
                subset = chosen[chosen.has_base] if name == "keeps_base" else chosen
                diff = (subset[f"ep_{name}"] - subset[f"et_{name}"]).to_numpy()
                if diff.size == 0:
                    continue
                sums = clustered_bootstrap(
                    subset.mol_id.to_numpy(), {"d": diff, "n": np.ones(diff.size)}, args.reps, 0
                )
                lo, hi = np.percentile(sums[:, 0] / sums[:, 1], [2.5, 97.5])
                calibration_rows.append(
                    {
                        "split": split,
                        "stratum": f"{stratum}={level}",
                        "attribute": name,
                        "pred": subset[f"ep_{name}"].mean(),
                        "target": subset[f"et_{name}"].mean(),
                        "diff": diff.mean(),
                        "lo": lo,
                        "hi": hi,
                        "ambiguous_target_share": subset[f"amb_{name}"].mean(),
                        "n": diff.size,
                    }
                )
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(output_dp / "calibration_in_the_large.csv", index=False)
    emit(
        "Table 6a. Calibration in the large: predicted - target attribute mean (all spectra)",
        calibration[calibration.stratum == "all=all"]
        .set_index(["attribute", "split"])
        .drop(columns=["stratum"])
        .round(4),
    )
    emit(
        "Table 6b. Predicted - target attribute mean by stratum, val",
        calibration[(calibration.split == "val") & (calibration.stratum != "all=all")]
        .pivot(index="stratum", columns="attribute", values="diff")
        .round(4),
    )

    # 9. Table 7: cross-fitted reweighting of the saved candidates.
    reweight_rows, coefficient_rows = [], []
    for path in args.inputs:
        loaded = np.load(Path(path).with_suffix(".cands.npz"))
        arrays = {
            "log_pred": np.log(np.maximum(loaded["pred"].astype(float), 1e-12)),
            "tgt": loaded["tgt"].astype(float),
            "x": loaded["x"].astype(float),
            "offsets": loaded["offsets"],
            "target_norm": loaded["target_norm"],
        }
        attribute_names = [str(a) for a in loaded["attributes"]]
        frame = pd.DataFrame({"spec_id": loaded["spec_id"]}).merge(
            spectra[["spec_id", "mol_id", "split", "nce_bin", "cos"]], on="spec_id", how="left"
        )
        split = frame.split.iloc[0]
        fold = (frame.mol_id.to_numpy() % 2).astype(int)
        raw, _ = reweighted_cosine(
            np.concatenate([[1.0], np.zeros(len(attribute_names))]),
            arrays["log_pred"],
            arrays["tgt"],
            arrays["x"],
            arrays["offsets"],
            arrays["target_norm"],
        )
        variants = {
            "gamma": (None, False),
            "gamma+attributes": (list(range(len(attribute_names))), False),
            "gamma+attributes per NCE bin": (list(range(len(attribute_names))), True),
        }
        held_out = {name: np.full(raw.size, np.nan) for name in variants}
        for name, (columns, by_nce) in variants.items():
            groups = frame.nce_bin.astype(str).to_numpy() if by_nce else np.zeros(raw.size, str)
            for group in np.unique(groups):
                for k in (0, 1):
                    train_rows = np.flatnonzero((fold == k) & (groups == group))
                    test_rows = np.flatnonzero((fold != k) & (groups == group))
                    theta = fit_reweighting(arrays, train_rows, columns)
                    test = _subset(arrays, test_rows)
                    held_out[name][test_rows] = reweighted_cosine(
                        theta,
                        test["log_pred"],
                        test["tgt"],
                        test["x"],
                        test["offsets"],
                        test["target_norm"],
                    )[0]
            if not by_nce:
                theta = fit_reweighting(arrays, np.arange(raw.size), columns)
                coefficient_rows.append(
                    {"split": split, "variant": name, "gamma": theta[0]}
                    | dict(zip(attribute_names, theta[1:], strict=True))
                )
        row = {
            "split": split,
            "spectra": raw.size,
            "raw": raw.mean(),
            "stage1_cos": frame.cos.mean(),
        }
        for name, values in held_out.items():
            gain = values - raw
            sums = clustered_bootstrap(
                frame.mol_id.to_numpy(), {"g": gain, "n": np.ones(gain.size)}, args.reps, 0
            )
            lo, hi = np.percentile(sums[:, 0] / sums[:, 1], [2.5, 97.5])
            row[name] = values.mean()
            row[f"{name} gain"] = gain.mean()
            row[f"{name} gain 95%"] = f"[{lo:+.4f}, {hi:+.4f}]"
        reweight_rows.append(row)
    emit(
        "Table 7a. Cross-fitted reweighting, held-out mean cosine",
        pd.DataFrame(reweight_rows).set_index("split").T,
    )
    emit(
        "Table 7b. Reweighting fitted on the whole split (log-weight per unit)",
        pd.DataFrame(coefficient_rows).set_index(["split", "variant"]).round(3),
    )

    # 10. Table 8: pairwise accuracy by how the measured #1 and #2 relate.
    pair_names = {
        "first_inside_second": "nested_winner_deeper",
        "second_inside_first": "nested_winner_shallower",
    }
    pair["relation"] = [
        pair_names.get(r, r)
        for r in map(pair_relation, pair.atoms_1, pair.atoms_2, pair.mol_atoms_1)
    ]
    pair["unambiguous"] = (pair.n_structures_1 == 1) & (pair.n_structures_2 == 1)
    for label, subset in (("all pairs", pair), ("one structure per peak", pair[pair.unambiguous])):
        table = (
            subset.groupby(["relation", "split"]).correct.agg(n="size", acc="mean").unstack("split")
        )
        if ("acc", "train") in table and ("acc", "val") in table:
            table[("deficit", "val-train")] = table[("acc", "val")] - table[("acc", "train")]
        table[("share", "val")] = table[("n", "val")] / table[("n", "val")].sum()
        emit(f"Table 8a. Pair accuracy by relation, {label}", table.round(3))
    nested = pair[pair.relation.str.startswith("nested")].copy()
    nested["winner_deeper"] = nested.relation == "nested_winner_deeper"
    for column in ("nce_bin", "novelty_bin"):
        table = (
            nested.groupby([column, "split", "winner_deeper"], observed=True)
            .correct.mean()
            .unstack("winner_deeper")
        )
        table.columns = ["acc_winner_shallower", "acc_winner_deeper"]
        table["balanced"] = table.mean(axis=1)
        table["asymmetry"] = table.acc_winner_shallower - table.acc_winner_deeper
        emit(f"Table 8b. Nested pairs by {column}", table.unstack("split").round(3))
    deeper = np.where(nested.winner_deeper, nested.loss_formula_1, nested.loss_formula_2)
    shallower = np.where(nested.winner_deeper, nested.loss_formula_2, nested.loss_formula_1)
    nested["step"] = [formula_difference(d, h) for d, h in zip(deeper, shallower, strict=True)]
    frequent_steps = nested[nested.split == "val"].step.value_counts().head(14).index
    table = (
        nested[nested.step.isin(frequent_steps)]
        .groupby(["step", "split", "winner_deeper"])
        .correct.mean()
        .unstack("winner_deeper")
    )
    table.columns = ["acc_winner_shallower", "acc_winner_deeper"]
    table["balanced"] = table.mean(axis=1)
    table["n"] = nested[nested.step.isin(frequent_steps)].groupby(["step", "split"]).size()
    emit(
        "Table 8c. Nested pairs by the step the deeper candidate takes",
        table.unstack("split").round(3),
    )
    (output_dp / "tables.txt").write_text("".join(out_lines))


if __name__ == "__main__":
    main()
