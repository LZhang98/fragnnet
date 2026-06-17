# DAG Missing Peak Findings — [M-H]⁻ Negative Mode

Analysis of `dag_missing_by_class_m-h.txt` (FT, [M-H]-, min_intensity=0.05,
min_spectra=20). Source script: `scripts/eval/find_missing_peaks.py --frag_dp
data/frag/nist23_ma_mi_d3_h4_isoFalse.tar`.

---

## Cholestane steroids (n=32, DAG miss rate=100.0%)

**17 mols; 3 in tar (mol_ids 5965, 6773, 6778).** Tetracyclic steroid backbone
(cholestane skeleton); mols include free hydroxyl steroids, an epoxide steroid,
and one cholesteryl sulfate.

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 466.3   | 2      | 0.289     | ¹³C isotope of cholesteryl sulfate [M-H]⁻=465.3 |

### In DAG but model underpredicts (category B)

*(none reported)*

### Root cause

**Artifact (tar coverage) dominant; direct anion secondary.**

14 of 17 mols (22 FT [M-H]⁻ spectra) are absent from the fragmentation tar and
contribute to the 100% miss rate in the header count. The 3 in-tar mols include
mol_id=6778 (cholesteryl sulfate, C₂₇H₄₆O₄S, [M-H]⁻=465.3), whose measured
spectra are dominated by [HSO₄]⁻ (96.96) and SO₃⁻• (79.96) — direct sulfate
anions that BFS cannot model because no bond cut on the steroid skeleton produces
these fragments. The lone category A entry (466.3, n=2) is the ¹³C isotope peak
of the [M-H]⁻ precursor, not a genuine fragment, and its presence in the MS2
spectrum is a data quality artifact (co-isolated precursor isotope window).

---

## Flavonoids (n=24, DAG miss rate=84.7%)

**27 mols; 23 in tar.** Synthetic and natural flavone aglycones, many lacking
hydroxyl substituents; in-tar set includes halogenated variants (6-bromoflavone
mol_id=50271, two fluoroflavones mol_ids=50166/50272), simple flavone (50278),
and ester/extended-ring flavones.

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 223.0   | 6      | 0.731     | above precursor (flavone [M-H]⁻=221.06) → co-isolation contamination |
| 221.2   | 6      | 0.194     | also above/at precursor; co-isolation |
| 220.1   | 6      | 0.199     | co-isolation variant |
| 111.0   | 6      | 0.258     | exact 110.957 in spectra → sulfate series or Br-cascade |
| 143.9   | 8      | 0.073     | Br-retaining fragment from 6-bromoflavone (50271) |
| 158.9   | 8      | 0.077     | Br-retaining fragment from 6-bromoflavone |
| 174.9   | 6      | 0.083     | Br-retaining fragment from 6-bromoflavone |
| 175.9   | 6      | 0.116     | Br-retaining fragment from 6-bromoflavone |
| 64.0    | 8      | 0.204     | exact 63.961 = SO₂⁻• from sulfate mobile-phase contamination |
| 79.0    | 8      | 0.172     | [C₅H₃O]⁻ (79.02) chromone ring fragment, or [PO₃]⁻ phosphate contamination |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 149.0   | 4      | 0.174          | 0.000          | [M-H-CO-CO₂]⁻ from flavone |
| 93.0    | 4      | 0.202          | 0.000          | phenolate [C₆H₅O]⁻ — model weakness |
| 107.0   | 2      | 0.151          | 0.010          | |
| 63.0    | 2      | 0.208          | 0.033          | |

### Root cause

**Data quality dominant; direct anion secondary.**

Two n=nan-CE spectra of flavone itself (mol_id=50278) have peaks at 63.96, 79.96,
96.96, and 110.96 — matching SO₂⁻•, SO₃⁻•, HSO₄⁻, and an unassigned sulfate
fragment — accounting for the 64.0/79.0 category A entries (8 spectra). A peak at
223.028 appears above the [M-H]⁻=221.060 precursor (confirmed from actual spectra),
indicating co-isolated compound contamination. These n=nan spectra appear to have
been acquired under sulfate-buffered or contaminated conditions. The .9 fractional
bins (143.9, 158.9, 174.9, 175.9) are Br-retaining direct anion fragments from
6-bromoflavone (50271), inaccessible to BFS. Category B: 93.0 = phenolate
[C₆H₅O]⁻ is in the DAG but model predicts 0 (model weakness).

---

## Pyrenes (n=65, DAG miss rate=82.8%)

**13 mols; 7 in tar.** PAH pyrene ring system (C₁₆H₁₀ core) with diverse
substituents: hydroxyl (51734), carboxylate (46096), mono/tri/tetra-sulfonate
(49004, 48933, 48942), dione (49983), iodoacetyl-amino (42473). The 3 polysulfonate
pyrenes dominate the spectra count.

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 233.1   | 37     | 0.445     | [M-H-2SO₃-SO₂]⁻ from 48933 (confirmed: 233.060) |
| 217.1   | 22     | 0.735     | [M-H-3SO₃-SO₂]⁻ from 48942 |
| 189.1   | 20     | 0.638     | [M-H-CO]⁻ from 1-hydroxypyrene 51734 (189.070) |
| 234.1   | 18     | 0.103     | ¹³C isotope of 233.1 — data quality artifact |
| 232.1   | 18     | 0.396     | [M-H-2SO₃-SO₂] variant or unassigned |
| 313.0   | 13     | 0.076     | [M-H-SO₃-SO₂]⁻ from 48933 (313.017 confirmed) |
| 298.0   | 12     | 0.123     | SO₃/SO₂ cascade fragment, unassigned |
| 190.1   | 12     | 0.123     | ¹³C isotope of 189.1 — data quality artifact |
| 457.9   | 9      | 0.132     | ¹³C isotope of 48933 [M-H]⁻=456.935 → 457.938 — data quality artifact |
| 378.0   | 9      | 0.072     | SO₃ cascade fragment, unassigned |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 217.1   | 13     | 0.117          | 0.006          | also in cat A; model weakness |
| 520.9   | 2      | 0.397          | 0.017          | [M-H]⁻ of 48942 (520.948) in MS2 — precursor retention |
| 281.0   | 4      | 0.195          | 0.016          | [M-H]⁻ of 49004 (281.041) or [M-H-3SO₃]⁻ from 48942 |
| 297.0   | 3      | 0.131          | 0.018          | [M-H-2SO₃]⁻ from 48933 (297.022 confirmed) |
| 456.9   | 1      | 0.395          | 0.010          | [M-H]⁻ of 48933 (456.935) — precursor retention |

### Root cause

**Rearrangement (SO₃/SO₂ cascade from polysulfonate pyrenes) + data quality
(¹³C isotope peaks) dominant.**

The trisulfonyl-hydroxypyrene (48933, [M-H]⁻=456.935) and tetrasulfonylpyrene
(48942, [M-H]⁻=520.948) each produce a cascade of SO₃ and SO₂ neutral losses in
negative mode. Confirmed fragment series from 48933: [M-H-SO₃]⁻=377.0,
[M-H-SO₃-SO₂]⁻=313.0, [M-H-2SO₃]⁻=297.0, [M-H-2SO₃-SO₂]⁻=233.1. These are
all rearrangement products — BFS makes no bond cuts on the sulfonate group and
cannot produce SO₂ or SO₃ neutral losses. The ¹³C isotope peaks of 48933 [M-H]⁻
(457.9) and of the 233.1 fragment (234.1) appear as spurious category A entries
(MS2 co-isolation of the M+1 isotope). From 1-hydroxypyrene (51734), [M-H-CO]⁻
=189.1 (CO loss from phenolate anion) is a known rearrangement absent from BFS.
Six missing mols (including both diaminopyrenes and both nitropyenes) inflate the
n=65 header but their spectra are skipped.

---

## Summary

| Subclass | n | miss% | Root cause summary |
|----------|---|-------|--------------------|
| Cholestane steroids | 32 | 100.0% | **Artifact** (14/17 mols not in tar) + **direct anion** (HSO₄⁻, SO₃⁻• from cholesteryl sulfate 6778); 466.3 = ¹³C precursor isotope in MS2 |
| Flavonoids | 24 | 84.7% | **Data quality** (sulfate/phosphate mobile-phase contamination in n=nan-CE flavone spectra; 223.0 > precursor 221.06 → co-isolation) + **direct anion** (Br-retaining .9 bins from 6-bromoflavone 50271) |
| Pyrenes | 65 | 82.8% | **Rearrangement** (SO₃/SO₂ cascade: 233.1, 217.1, 313.0 from polysulfonate pyrenes; CO loss 189.1 from hydroxypyrene) + **data quality** (¹³C isotope peaks 457.9, 234.1, 190.1 appearing as missing fragments) |
