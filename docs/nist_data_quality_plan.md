# NIST Data Quality Analysis Plan

Analysis to be added to `notebooks/frag_gen_comparison.ipynb` after the existing low-wrecall section (cell 29).

## Current notebook coverage (cells 21–29)

Quality tiers (binned on `wrecall_0.01_h4`) broken down by:
- `inst_type`, `frag_mode`, `prec_type` — stacked bar % (cell 24)
- `num_peaks` distribution by tier (cell 25)
- ACE (QTOF only) / NCE (FT only) CE distribution by tier, % (cell 26)
- Molecule-specific vs spectrum-specific bad spectra (cells 27–28)

Root causes identified so far: QTOF high-CE, CID mode, [M-H]−, few peaks, ~20 all-bad molecules.

---

## Proposed new section: NIST Data Quality + Formula Decomposition

### A. Mol metadata join

Join `mol_df` (`formula`, `exact_mw`, `num_atoms`) into `merged` on `mol_id`.
Prerequisite for sections B and E.

```python
PROC_DP = Path("../data/proc/nist23")
mol_df = pd.read_pickle(PROC_DP / "mol_df.pkl")[["mol_id", "formula", "exact_mw", "num_atoms"]]
merged = merged.merge(mol_df, on="mol_id", how="left")
```

### B. Formula decomposition of peaks

For each spectrum, load its peaks from `spec_df`, then for each peak m/z:

1. Compute neutral mass = `peak_mz − adduct_offset` (derived from `prec_type`)
2. Check if neutral mass ≤ precursor mass and the mass difference is decomposable into
   a valid subformula within the element bounds of the precursor formula (C/H/N/O/S/P)
3. Output `frac_peaks_explainable` per spectrum

This is independent of the BFS DAG — it answers:
> *"Is this peak chemically possible from this precursor at all?"*

Common adduct offsets needed:

| prec_type | adduct offset (Da) |
|-----------|-------------------|
| [M+H]+    | +1.00728          |
| [M-H]-    | −1.00728          |
| [M+Na]+   | +22.98922         |
| [M+NH4]+  | +18.03437         |

### C. DAG gap vs spectrum noise quadrant

Cross-plot `wrecall_h2` (DAG coverage) vs `frac_peaks_explainable` (formula coverage):

|                     | High formula-explainable | Low formula-explainable |
|---------------------|--------------------------|-------------------------|
| **High wrecall**    | ✅ Model working          | (rare)                  |
| **Low wrecall**     | 🔧 DAG gap — fixable      | ❌ Noisy spectrum — NIST issue |

Quantify each quadrant to separate "what the model can improve" from "bad NIST entries to flag".

### D. Intrinsic spectrum quality flags

No model needed — computable directly from peaks:

- **Precursor peak present**: any peak ≈ `exact_mw + adduct_offset` within ±0.01 Da
  (unfragmented precursor leakthrough — indicates poor fragmentation)
- **Peaks above precursor**: count of peaks > precursor m/z + 2 Da
  (contamination or wrong precursor assignment)
- **Intensity entropy**: `−Σ p·log(p)` over normalized peak intensities
  (low entropy = spectrum dominated by 1–2 peaks, often low quality)

Show all three flags broken down by quality tier as stacked bar or violin plots.

### E. Molecular complexity vs wrecall

Using `num_atoms` and ring count (via RDKit on `formula` or `mol`):

- Scatter: `num_atoms` vs median `wrecall_h2` per molecule
- Bin by ring count: do polycyclic molecules have lower wrecall?
- Identifies structural blind spots in fragmentation coverage

---

## Implementation notes

- Section B requires reloading `peaks` from `spec_df` (1.9M rows, ~large).
  Consider sampling 50k spectra or processing in batches.
- Formula validity check: parse precursor formula with `re` into element counts,
  then check candidate neutral formula element-wise ≤ precursor counts.
- `spec_stats_df` is freed after cell 29 — sections B–E work from `merged` + fresh `spec_df` load.
- Implementation order: **A → B → C** (most novel), then **D** (quick wins), then **E**.
