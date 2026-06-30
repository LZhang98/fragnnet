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
been acquired under sulfate-buffered or contaminated conditions. The .9 fractional bins (143.9, 158.9, 174.9, 175.9) are Br-retaining
fragments from 6-bromoflavone (50271). The most likely pathway is retro-Diels-Alder
(RDA) on the chromone C ring (concerted, even-electron), which retains Br on the
A-ring fragment. BFS supports 2-bond ring cuts but misses this pathway because both
RDA cut bonds (C2–C3 and C4–C4a) are in the same ring of the fused A+C bicyclic
system — BFS does not correctly enumerate this bond pair for fused bicyclics where
the shared C4a–C8a bond must remain intact. Category B: 93.0 = phenolate
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

## Pyrimidodiazepines (n=30, DAG miss rate=73.3%)

**6 mols; 2 in tar (21531, 25475).** Dibenz-fused diazepinedione structures (pyrimido-diazepine
ring systems), including drug-like fluorinated pyrimidine-diazepinone derivatives.

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 59.0    | 12     | 1.000     | 59.012 = CH₃COO⁻ (acetate) — mobile phase contamination |
| 97.0    | 12     | 0.341     | 96.959 = H₂PO₄⁻ — phosphate mobile phase contamination |
| 99.9    | 12     | 0.257     | 99.924 — unidentified .9-bin, possibly contamination |
| 100.9   | 12     | 0.054     | 100.9 — adjacent to 99.9, possibly contaminant isotope pair |
| 115.9   | 12     | 0.208     | 115.919 — unidentified .9-bin; 115.9 − 99.9 = 16.0 (O shift) |
| 116.9   | 12     | 0.108     | 116.9 — adjacent .9-bin pair with 115.9 |
| 130.9   | 12     | 0.096     | 130.9 — next member of .9-bin series |
| 309.1   | 12     | 0.270     | Genuine fragment from mol 21531 ([M-H]⁻=394.12); neutral loss 85.0 |
| 311.2   | 12     | 0.338     | Adjacent to 309.1 (Δ=2.07 Da); appears in 2 of 3 spectra of 21531 |
| 395.2   | 12     | 0.140     | 395.389 in 21531 spectrum — above precursor 394.12 → co-isolation |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 62.0    | 12     | 0.351          | 0.000          | 61.987 = NO₃⁻ — nitrate contamination |
| 74.0    | 12     | 0.235          | 0.000          | possible small fragment from 21531 or contaminant |
| 89.0    | 12     | 0.311          | 0.010          | genuine fragment from 21531 (89.023) |
| 191.0   | 12     | 0.084          | 0.000          | fragment from 21531 or contamination |

### Root cause

**Data quality dominant** (nce=nan contaminated spectra from the only in-tar mol 21531).

Mol 21531 (C₂₄H₁₅N₂O₄⁻?, [M-H]⁻=394.12) has only 3 FT [M-H]⁻ spectra, all with nce=nan. In
all 3, the base peak is 59.012 (CH₃COO⁻, acetate), with 96.959 (H₂PO₄⁻) and 61.987 (NO₃⁻) as
major peaks — unambiguous mobile-phase buffer contamination. The .9-bin series (99.9/115.9/116.9
/130.9) with 16.0 Da spacings is consistent with an oxidation series from an unidentified
contaminant or a very minor fragment from 21531. The above-precursor peak at 395.389 (bin 395.2)
in the 3rd spectrum confirms co-isolated compound contamination. The n=12 header count vs. only
3 analyzable spectra suggests the analysis script may be counting spectra from mol 25476 (not in
tar) or additional unlisted classyfire entries, but the root cause is the same: all in-tar spectra
are contaminated nce=nan acquisitions.

---

## Sulfinylbenzimidazoles (n=26, DAG miss rate=71.4%)

**21 mols; 15 in tar.** Proton-pump inhibitor scaffold and related compounds: benzimidazoles with
sulfinyl, sulfonyl, or sulfonyloxy substituents at the C2 position; many are thioether/sulfoxide
(S⁺([O⁻])) zwitterions.

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 132.0   | 7      | 0.832     | 132.032 = rearrangement product from sulfonylbenzimidazoles (41891, 7436, 30773) |
| 132.1   | 7      | 0.611     | 132.056 = [C₇H₆N₃]⁻ (aminobenzimidazolide) from SO₂ loss in mol 40025 |
| 148.1   | 1      | 0.056     | minor fragment, n=1 |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 116.0   | 2      | 0.111          | 0.037          | [C₇H₄N₂]⁻ benzimidazol-2-yl from direct bond cut |
| 78.0    | 1      | 0.194          | 0.028          | SO₃⁻• or other small anion; model weakness |
| 64.0    | 1      | 0.055          | 0.000          | SO₂⁻• = 63.962, in DAG but model predicts 0 |
| 178.0   | 1      | 0.062          | 0.020          | minor, n=1 |
| 212.0   | 1      | 0.112          | 0.035          | minor, n=1 |

### Root cause

**Rearrangement (SO₂/SO₃ neutral loss from sulfonylbenzimidazoles) dominant.**

Mol 40025 (aminosulfonylbenzimidazole, [M-H]⁻=196.018) produces 132.056 = [C₇H₆N₃]⁻ =
deprotonated aminobenzimidazole, formed by SO₂ loss (Δ=63.962 Da): [M-H-SO₂]⁻ = 132.056. This
is a rearrangement; BFS does not model SO₂ extrusion from the sulfonyl group. Mol 41891
(aminobenzimidazole-2-sulfonic acid, [M-H]⁻=212.012) produces both 132.056 (SO₃ loss, Δ=79.957)
and 132.032 (a related rearrangement product). Mols 7436 and 30773 also show 132.032-133 across a
wide CE range. Category B peaks (116.0, 64.0) are direct BFS fragments but the model
under-predicts them; particularly 64.0 = SO₂⁻• is in the DAG but model assigns ≈0 probability.

---

## 9,9p-dihydroxyaryltetralin lignans (n=58, DAG miss rate=68.4%)

**5 mols; 5 in tar (25699, 25792, 25797, 25801, 26788).** Aryltetralin lignans: bicyclic
dihydronaphthalene (tetralin) ring with guaiacyl/syringyl aryl substituents and hydroxymethyl
groups at C7/C8; all 5 mols and all 65 spectra are fully analyzable.

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 196.1   | 22     | 0.486     | 196.052 = C₁₃H₈O₂⁻• (radical anion, exact); all 5 mols |
| 197.1   | 22     | 0.390     | 197.060 = [C₁₃H₁₀O₂-H]⁻; all 5 mols |
| 195.0   | 20     | 0.483     | 195.044 = C₁₃H₈O₂ related; all 5 mols |
| 183.0   | 18     | 0.519     | 183.044 = differs from 195.0 by 12.000 Da (C); all 5 mols |
| 209.1   | 16     | 0.209     | 209.1 series — 196.1 + 13 (CH) or 183+26 (2C) |
| 253.1   | 16     | 0.388     | two sub-series: 253.050 and 253.086 from different mols |
| 145.0   | 14     | 0.186     | 145.0 small aromatic fragment |
| 241.1   | 14     | 0.396     | 241.046 (all 5 mols) |
| 225.1   | 14     | 0.503     | 225.052-225.056 (all 5 mols); 225.1 − 197.1 = 28.0 = CO |
| 224.0   | 14     | 0.421     | adjacent to 225.1 |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 159.0   | 16     | 0.286          | 0.012          | guaiacyl-methylene fragment; BFS reachable but model weak |
| 329.1   | 13     | 0.118          | 0.010          | large ring-opening fragment, model weak |
| 181.1   | 12     | 0.169          | 0.002          | guaiacyl acrolein anion in DAG; model weak |
| 125.0   | 12     | 0.096          | 0.015          | small aromatic; model weak |
| 191.1   | 12     | 0.126          | 0.017          | C₁₁H₁₀O₃⁻ from single cut; model weak |
| 214.1   | 12     | 0.091          | 0.012          | |
| 239.0   | 10     | 0.202          | 0.017          | |
| 313.1   | 10     | 0.225          | 0.023          | |
| 299.1   | 10     | 0.175          | 0.026          | |
| 95.0    | 10     | 0.194          | 0.013          | |

### Root cause

**Rearrangement (retro-Diels-Alder / homolytic tetralin ring cleavage) dominant for cat A;
model weakness for cat B.**

All 10 category A peaks appear across all 5 structurally diverse aryltetralin mols with high
intensity (0.19–0.52), confirming systematic DAG coverage failure, not per-mol artifact. The
diagnostic fragment at 196.052 matches C₁₃H₈O₂⁻• (radical anion) exactly; the series 183.044 →
195.044 → 196.052 (separated by 12.000 and 1.008 Da) and the CO-loss series 253.1 → 225.1 →
197.1 (Δ=28.0 each) indicate aromatic rearrangement products from the tetralin ring system.
BFS cannot model the retro-Diels-Alder opening of the dihydronaphthalene core nor the subsequent
radical C–C homolysis that generates C₁₃ aromatic fragments. The category B peaks (181.1, 191.1,
etc.) are reachable by BFS as single-cut guaiacyl fragments, but the model systematically
underpredicts them — likely because the training loss averages over the dominant cat A peaks that
the model can never learn.

---

## Pyridine carboxaldehydes (n=44, DAG miss rate=66.7%)

**3 mols; 3 in tar.** Pyridoxal (vitamin B6 aldehyde) and phosphorylated derivatives; the dominant
mols are pyridoxal-5-phosphate (34154, [M-H]⁻=246.017) and a diazo-sulfonated derivative (34028).

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 247.0   | 6      | 0.113     | 247.022 = ¹³C isotope of [M-H]⁻=246.017 leaking into MS2; data quality |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 228.0   | 2      | 0.053          | 0.006          | minor fragment from 34154/34028 |
| 79.0    | 1      | 0.062          | 0.025          | 78.960 = PO₃⁻ in DAG but model underpredicts |

### Root cause

**Direct anion (phosphate head group) dominant; data quality secondary.**

Mol 34154 (pyridoxal-5-phosphate, [M-H]⁻=246.017) spectra are dominated by PO₃⁻ (78.960, base
peak), H₂PO₄⁻ (96.970), and HPO₂⁻ (62.965) — the phosphate moiety fragments to direct anions
that BFS cannot model (cutting both P-O ester bonds requires 2 cuts to release the phosphate
intact, but the resulting PO₃⁻/H₂PO₄⁻ is a direct inorganic anion, not a [neutral±H]⁻ BFS
product). The sole category A entry (247.022) is the ¹³C isotope of the [M-H]⁻ precursor
appearing in the MS2 isolation window — a data quality artifact.

---

## Aporphines (n=75, DAG miss rate=64.3%)

**51 mols; 19 in tar.** Tetracyclic isoquinoline alkaloids (dibenzo[de,g]quinoline skeleton)
with methoxy/methylenedioxy substituents; includes nantenine, predicentrine, boldine, and
N-methyl aporphines.

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 224.0   | 29     | 0.439     | 224.048 = C₁₄H₉O₃⁻ and 224.072 = C₁₅H₁₂O⁻ (two mol series); absent from DAG |
| 235.0   | 18     | 0.699     | 235.040/235.077 — two series; high intensity |
| 225.1   | 19     | 0.345     | adjacent to 224.0 |
| 207.0   | 19     | 0.343     | |
| 234.1   | 18     | 0.097     | |
| 223.0   | 18     | 0.190     | |
| 167.1   | 17     | 0.457     | |
| 196.1   | 16     | 0.177     | |
| 195.0   | 16     | 0.529     | |
| 209.0   | 15     | 0.342     | |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 265.1   | 53     | 0.701          | 0.007          | 265.050 = C₁₆H₉O₄⁻; in DAG, model ≈0 |
| 250.1   | 67     | 0.701          | 0.006          | 250.050/250.064; in DAG, model ≈0 |
| 191.1   | 19     | 0.580          | 0.000          | in DAG, model 0 |
| 163.1   | 23     | 0.602          | 0.001          | in DAG, model ≈0 |
| 181.1   | 13     | 0.166          | 0.002          | |
| 266.1   | 13     | 0.144          | 0.009          | |

### Root cause

**Model weakness dominant; cascade depth secondary.**

The category B peaks are the most damaging miss: 265.1 (C₁₆H₉O₄⁻, 53 spectra, mean intensity
0.701) and 250.1 (67 spectra, 0.701) are both reachable by BFS but the model assigns near-zero
probability. Similarly, 163.1 (0.602) and 191.1 (0.580) are in DAG but model predicts ≈0. These
fragments span 8 different mols, confirming systematic model failure on the aporphine ring system.
Category A (224.0 absent from DAG across 29 spectra) additionally points to cascade depth —
C₁₄H₉O₃⁻ and C₁₅H₁₂O⁻ fragments require loss of the nitrogen-containing ring portion, which
likely involves rearrangement or ≥3 bond cuts from the tetracyclic precursor.

---

## Tropane alkaloids (n=22, DAG miss rate=63.7%)

**24 mols; 19 in tar.** Tropane ester alkaloids including ecgonine derivatives, cocaine analogs,
and hydroxytropane esters; one key mol is 21573 (N-methyl-3α-phenacyloxytropane type,
[M-H]⁻=288.160).

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 62.0    | 9      | 0.266     | 61.987 = NO₃⁻ — mobile phase contamination (mol 21573, nce=nan) |
| 270.9   | 7      | 0.218     | 270.922 — .9-bin contaminant; above-precursor 311.169 in same spectra confirms co-isolation |
| 115.9   | 7      | 0.148     | 115.919 — .9-bin background contaminant (also in pyrimidodiazepines, furanocoumarins) |
| 116.9   | 7      | 0.121     | 116.927 — .9-bin pair with 115.9 |
| 97.0    | 6      | 0.369     | 96.959 = H₂PO₄⁻ — phosphate contamination (mol 21573, nce=nan) |
| 99.9    | 6      | 0.086     | 99.924 — recurrent .9-bin background contaminant |
| 207.0   | 6      | 0.237     | minor fragment from contaminated spectra |
| 80.0    | 5      | 0.102     | 79.956 = SO₃⁻• — sulfate contamination |
| 204.0   | 4      | 0.082     | minor |
| 287.0   | 4      | 0.248     | minor, possibly genuine fragment from mol 21573 |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 365.2   | 13     | 0.533          | 0.015          | in DAG, model ≈0 — model weakness |
| 59.0    | 9      | 0.642          | 0.004          | 59.012 = acetate in contaminated spectra (nce=nan) OR tropane ester fragment |
| 508.3   | 8      | 0.523          | 0.013          | large precursor-like fragment, in DAG, model ≈0 |
| 131.1   | 3      | 0.466          | 0.003          | |
| 74.0    | 3      | 0.552          | 0.000          | |

### Root cause

**Data quality dominant (cat A); model weakness secondary (cat B).**

All 9 category A .9-bin peaks (115.9/116.9/99.9/270.9) and the phosphate/nitrate/sulfate ions
originate from mol 21573 whose FT [M-H]⁻ spectra all have nce=nan. The sample spectrum shows
stearate 283.264 as base peak, with NO₃⁻ (61.987), H₂PO₄⁻ (96.959), CH₃COO⁻ (59.013), and
above-precursor ion at 311.169 (> prec 288.160 → co-isolation confirmed). The .9-bin series
(99.9/115.9/116.9/270.9) are recurrent background contaminants in nce=nan acquisitions across
multiple subclasses, not tropane-specific. Category B: 365.2 and 508.3 are genuine tropane
fragments in DAG with ~0.52-0.53 intensity but model predicts ≈0 (model weakness).

---

## Furanocoumarins (n=120, DAG miss rate=59.0%)

**89 mols; 73 in tar.** Benzofuran-fused pyranone natural products (psoralen, bergapten,
xanthotoxin, isopsoralen, and related) with diverse substituents; one of the most heavily
contaminated subclasses.

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 97.0    | 90     | 0.272     | 96.959 = H₂PO₄⁻ across 26 mols (nan=True); phosphate contamination |
| 62.0    | 78     | 0.698     | 61.987 = NO₃⁻ across 26 mols (nan=True) |
| 95.0    | 27     | 0.150     | probably PO₃-type or organic fragment; nan=True |
| 80.0    | 79     | 0.220     | 79.957 = SO₃⁻• across 26 mols (nan=True) |
| 79.0    | 93     | 0.105     | 78.958 = PO₃⁻/SO₃ across 28 mols (nan=True) |
| 64.0    | 74     | 0.134     | 63.961 = SO₂⁻• across 26 mols (nan=True) |
| 283.3   | 28     | 0.482     | 283.264 = stearate [C₁₈H₃₅O₂]⁻ — fatty acid contamination; 13 mols (nan=True) |
| 81.0    | 18     | 0.084     | 80–81 range; nan=True |
| 115.9   | 47     | 0.363     | .9-bin background contaminant; 19 mols (nan=True) |
| 116.9   | 49     | 0.156     | .9-bin pair with 115.9; 20 mols (nan=True) |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 183.0   | 28     | 0.281          | 0.002          | genuine furanocoumarin fragment; model weak |
| 89.0    | 26     | 0.268          | 0.012          | |
| 149.0   | 17     | 0.314          | 0.009          | |
| 211.0   | 12     | 0.505          | 0.012          | high-intensity genuine fragment; model weak |
| 73.0    | 20     | 0.162          | 0.022          | |

### Root cause

**Data quality overwhelmingly dominant; model weakness secondary for genuine fragments.**

The cat A peaks are the same mobile phase contaminants seen across multiple subclasses:
H₂PO₄⁻ (97.0, n=90, 26 mols), NO₃⁻ (62.0, n=78), SO₃⁻• (80.0, n=79), SO₂⁻• (64.0, n=74),
PO₃⁻ (79.0, n=93), stearate (283.3, n=28), and recurrent .9-bin contaminants (115.9, n=47).
All have nan=True and span 13-28 mols each — this is systematic nce=nan acquisition
contamination, not furanocoumarin-specific fragmentation. Category B (183.0/89.0/149.0/211.0)
contains real furanocoumarin fragments in the DAG that the model systematically underpredicts
(model weakness), but these are dwarfed by the contamination signal.

---

## Isoindoles (n=73, DAG miss rate=59.0%)

**41 mols; 39 in tar.** Benzo[c]pyrrole-based compounds including phthalimides, isoindolinones,
thalidomide analogs, and isoindoline derivatives; two key mols are the phthalimide prec=549.16
and an aromatic isoindole at prec=383.23.

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 282.1   | 22     | 0.342     | 282.080 from prec=549.16 only; large fragment absent from DAG |
| 280.1   | 18     | 0.168     | |
| 281.1   | 18     | 0.222     | |
| 254.1   | 16     | 0.513     | 254.085 from prec=549.16; cascade depth from large mol |
| 140.1   | 16     | 0.701     | 140.050/140.051 from two mols |
| 164.1   | 14     | 0.160     | |
| 206.0   | 14     | 0.207     | |
| 178.1   | 14     | 0.254     | |
| 205.1   | 12     | 0.228     | |
| 152.1   | 12     | 0.346     | 152.050/152.051 |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 108.0   | 12     | 1.000          | 0.001          | BASE PEAK in DAG, model predicts 0.001 — severe model weakness |
| 166.1   | 7      | 0.879          | 0.000          | 166.050/166.066; very high intensity, model predicts 0 |
| 69.0    | 12     | 0.733          | 0.003          | in DAG, model weak |
| 311.1   | 8      | 0.210          | 0.019          | |
| 161.0   | 6      | 0.180          | 0.032          | |

### Root cause

**Model weakness dominant (cat B); cascade depth secondary (cat A).**

Category B contains a spectacular failure: 108.0 (n=12 spectra, mean intensity=1.000 = base peak
across all spectra) is reachable by BFS but model predicts 0.001. Similarly 166.1 (intensity 0.879,
n=7, model 0.000) and 69.0 (intensity 0.733, n=12, model 0.003). These fragments span multiple
isoindole mols and occur at high CE where fragmentation is complete, but the model assigns
near-zero probability to the base peak. Category A (282.1/280.1/254.1 from prec=549.16) points
to cascade depth: the large isoindole compound (mol with [M-H]⁻=549.16) produces mid-sized
fragments that require ≥4 bond cuts from the intact parent.

---

## Lignan glycosides (n=379, DAG miss rate=58.3%)

**54 mols; 41 in tar.** Glycosylated lignans: aryltetralin, furofuran, and dibenzylbutyrolactone
lignans with mono/di/tri-saccharide units; precursor masses span 491–921 Da.

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 166.0   | 117    | 0.368     | 166.026 = C₈H₆O₄⁻• (or phthalate radical); 24 mols, nan=True |
| 136.0   | 242    | 0.524     | 136.015 — unknown; 34 mols, nan=True |
| 151.0   | 271    | 0.662     | 151.002/151.003; 34 mols, nan=True; extremely high n |
| 212.0   | 15     | 0.199     | |
| 261.1   | 16     | 0.101     | |
| 219.9   | 14     | 0.556     | .9-bin; 5 mols, nan=True — halide contamination |
| 209.1   | 14     | 0.233     | |
| 169.1   | 14     | 0.111     | |
| 197.1   | 13     | 0.191     | 197.060 = C₁₃ aromatic rearrangement (same as aryltetralin lignans) |
| 196.1   | 114    | 0.232     | 196.052 = C₁₃H₈O₂⁻• radical — same rearrangement motif as aryltetralin lignans |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 145.0   | 46     | 0.240          | 0.003          | in DAG, model weak |
| 279.1   | 36     | 0.173          | 0.001          | |
| 153.1   | 31     | 0.300          | 0.008          | |
| 225.1   | 29     | 0.304          | 0.005          | |
| 297.1   | 29     | 0.256          | 0.019          | |

### Root cause

**Data quality dominant for highest-n cat A peaks; rearrangement for 196.1/197.1.**

The top-3 cat A peaks by n (151.0=271, 136.0=242, 166.0=117) all have nan=True across 24-34 mols,
confirming systematic nce=nan contamination at scale. The sample nce=nan spectrum (mol 11453,
prec=921.303) shows background ions at 68.997, 93.033, 71.012 (likely sugar or mobile-phase
fragments). Peak 219.9 (n=14, nan=True) is a .9-bin halide contamination fragment. The 196.1
(n=114, 196.052=C₁₃H₈O₂⁻•) and 197.1 bins reproduce exactly the aryltetralin lignan C₁₃
radical anion pattern — these glycosidic lignans retain the aryltetralin aglycone which undergoes
the same retro-Diels-Alder/homolytic cleavage. Category B model weakness is also significant
(145.0, 153.1, 225.1 across 29-46 spectra).

---

## Delta valerolactones (n=35, DAG miss rate=57.0%)

**55 mols; 37 in tar.** Six-membered ring lactones (δ-valerolactone skeleton) with aryl, acyl,
and hydroxyl substituents; precursor range 129–471 Da. All key cat A spectra are nce=nan.

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 79.0    | 24     | 0.128     | 78.958/78.985 = PO₃⁻/SO₃-type; nan=True across 6 mols |
| 62.0    | 24     | 0.112     | 61.987 = NO₃⁻; nan=True |
| 95.0    | 18     | 0.316     | minor; nan=True |
| 80.0    | 19     | 0.212     | 79.956 = SO₃⁻•; nan=True |
| 97.0    | 33     | 0.273     | 96.959 = H₂PO₄⁻; nan=True (8 mols) |
| 64.0    | 18     | 0.113     | 63.961 = SO₂⁻•; nan=True |
| 81.0    | 12     | 0.143     | |
| 115.9   | 21     | 0.127     | .9-bin background contaminant; nan=True |
| 183.0   | 7      | 0.674     | 183.011 = unassigned; base peak in some nce=nan spectra; possibly genuine |
| 184.0   | 7      | 0.060     | |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 191.1   | 9      | 0.686          | 0.001          | in DAG, major model weakness |
| 149.0   | 14     | 0.066          | 0.008          | |
| 93.0    | 11     | 0.112          | 0.007          | |
| 223.0   | 9      | 0.120          | 0.000          | |
| 81.0    | 8      | 0.539          | 0.007          | |

### Root cause

**Data quality dominant (same nce=nan contaminant profile as furanocoumarins).**

Sample spectra (mol 1278 prec=471.239 nce=nan, mol 1898 prec=357.119 nce=nan) show the same
contaminant fingerprint: NO₃⁻ (61.987) as base peak or major peak, H₂PO₄⁻ (96.959), SO₃⁻•
(79.956), CH₃COO⁻ (59.012), and .9-bin background ions (115.919, 116.927, 99.924). The 97.0/62.0
/80.0/64.0/79.0 cat A peaks are all confirmed as inorganic mobile-phase contaminants. Peak 183.0
(7 spectra, high intensity 0.674) also appears in these nce=nan spectra; exact mass 183.011 is
unassigned but likely genuine (possibly a γ-ketosulfonate or phenyl-propionate from the lactone
skeleton). Category B: 191.1 (n=9, 0.686) in DAG but model predicts 0.001 — model weakness for
the real lactone ring-opening fragment.

---

## Anthracenes (n=121, DAG miss rate=55.8%)

**64 mols; 51 in tar.** Substituted anthracenes including hydroxy, carboxy, sulfonyl, and
nitro derivatives; the polycyclic aromatic system is represented by sulfoanthracenes (mol 1564
prec=261.067) and hydroxyanthracene carboxylates.

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 169.0   | 68     | 0.422     | 169.016/169.029 — two chemical series; 11 mols, nan=False (genuine) |
| 197.0   | 74     | 0.299     | 197.011/197.025/197.061 — three series; 12 mols |
| 153.0   | 52     | 0.258     | 153.022/153.034 — two series; 10 mols, nan=False |
| 181.0   | 23     | 0.182     | genuine aromatic fragment; nan=False |
| 180.0   | 20     | 0.176     | |
| 226.0   | 51     | 0.106     | 226.001/226.014; nan=True for some |
| 208.0   | 19     | 0.130     | |
| 141.0   | 57     | 0.563     | 140.996/141.022/141.034; 12 mols; high intensity |
| 125.0   | 18     | 0.289     | |
| 227.0   | 18     | 0.297     | |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 113.0   | 20     | 0.275          | 0.004          | in DAG, model weak |
| 343.0   | 15     | 0.379          | 0.007          | large aromatic fragment; in DAG, model weak |
| 89.0    | 16     | 0.116          | 0.007          | |
| 87.0    | 15     | 0.168          | 0.009          | |
| 167.0   | 14     | 0.224          | 0.010          | |

### Root cause

**Cascade depth dominant.**

All primary cat A peaks (169.0, 153.0, 141.0, 197.0) have nan=False for most contributing
spectra and span 10-12 diverse anthracene mols — these are genuine fragments absent from BFS,
not contamination. The tricyclic anthracene ring system (three fused 6-membered rings) requires
≥4 sequential bond cuts to produce C₉-C₁₃ aromatic fragments: first ring opening (+H-shift),
then further ring contraction, then small aromatic ion formation. The two mass series in each
bin (e.g., 169.016 and 169.029) correspond to different substituent patterns (sulfonated vs.
carboxylated anthracenes) both trapped by the same cascade depth bottleneck. Category B (343.0,
n=15, 0.379) represents large fragments accessible to BFS (1-2 cuts) but systematically
underpredicted by the model.

---

## Morphinans (n=87, DAG miss rate=54.8%)

**36 mols; 31 in tar.** Pentacyclic opioid alkaloids: morphine ([M-H]⁻=284.128), codeine
([M-H]⁻=298.108), naloxone, naltrexone, buprenorphine, and related; all key spectra have proper
CE (no nce=nan contamination).

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 197.1   | 39     | 0.264     | 197.060/197.097 — two series; 10 mols; absent from DAG |
| 185.1   | 34     | 0.477     | 185.024/185.060 — two series; ring-opening fragment absent from DAG |
| 147.0   | 18     | 0.126     | |
| 171.0   | 17     | 0.215     | |
| 169.1   | 16     | 0.189     | |
| 181.1   | 15     | 0.128     | |
| 119.1   | 14     | 0.247     | |
| 199.1   | 14     | 0.162     | |
| 159.0   | 14     | 0.126     | |
| 183.0   | 19     | 0.146     | |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 225.1   | 41     | 0.562          | 0.014          | 225.055/225.092 — two series; dominant fragment, model ≈0 |
| 283.1   | 16     | 0.723          | 0.006          | 283.084/283.120 — two series; in DAG, model barely predicts |
| 227.1   | 16     | 0.304          | 0.005          | |
| 226.1   | 18     | 0.240          | 0.009          | |
| 85.0    | 42     | 0.185          | 0.005          | most common cat B peak (42 spectra) |

### Root cause

**Model weakness dominant; cascade depth secondary.**

No nce=nan contamination — all failures are structural. Category B is the primary problem:
225.1 (n=41 spectra, intensity 0.562) and 283.1 (n=16, intensity 0.723) are both BFS-reachable
but model predicts 0.014/0.006. For codeine ([M-H]⁻=298.108), 283.120 = loss of 14.988 ≈ CH₃•
(N-demethylation radical loss) — a known morphinan fragmentation where the N-methyl cleaves
homolytically. Similarly 225.055 appears across 9 diverse mols (morphine, codeine, naloxone
series) at high intensity. Category A (185.1 n=34, 0.477; 197.1 n=39, 0.264 absent from DAG)
requires >3 cuts from the pentacyclic ring after the 4,5-epoxide bridge opens — cascade depth.

---

## Coumestans (n=24, DAG miss rate=54.8%)

**10 mols; 6 in tar.** Benzofuranbenzopyranone natural products: coumestrol ([M-H]⁻=267.030),
and methoxy/prenyl derivatives ([M-H]⁻=335.092 = C₂₀H₁₅O₅⁻).

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 195.0   | 10     | 0.516     | 195.044/195.046 from 3 mols; absent from DAG |
| 194.0   | 7      | 0.196     | adjacent to 195.0 |
| 182.0   | 6      | 0.123     | |
| 222.0   | 5      | 0.141     | |
| 169.0   | 5      | 0.419     | |
| 139.1   | 5      | 0.245     | |
| 154.0   | 4      | 0.142     | |
| 125.0   | 4      | 0.242     | |
| 153.0   | 4      | 0.182     | |
| 129.0   | 4      | 0.274     | |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 279.0   | 19     | 0.738          | 0.005          | 279.011/279.030 = C₁₆H₇O₅⁻ coumestan aglycone anion; model ≈0 |
| 223.0   | 16     | 0.527          | 0.022          | in DAG, model ≈0 |
| 266.0   | 14     | 0.492          | 0.015          | 266.022 |
| 179.1   | 6      | 0.156          | 0.000          | |
| 178.0   | 5      | 0.189          | 0.000          | |

### Root cause

**Model weakness dominant; rearrangement secondary.**

Category B contains the dominant failure: 279.0 (C₁₆H₇O₅⁻, 19 spectra, 0.738 intensity) is
BFS-reachable but model predicts 0.005. For the prenyl-coumestan (prec=335.092 = C₂₀H₁₅O₅⁻),
279.030 = [M-H - C₄H₈]⁻ = loss of prenyl as isobutylene (56.063 Da) — a direct BFS bond cut
from the O-prenyl ether. The model systematically assigns near-zero to this dominant fragment.
Similarly 223.0 (0.527) and 266.0 (0.492) are in DAG but predicted ≈0. Category A (195.0/194.0
n=10-15, absent from DAG) requires further ring-opening fragmentation of the coumestan skeleton
after the prenyl/substituent loss — likely retro-[4+2] or ring contraction rearrangement of
the fused benzofuran-chromene system, inaccessible to BFS.

---

## Strychnos alkaloids (n=22, DAG miss rate=53.7%)

**33 mols; 4 in tar.** Hexacyclic indole alkaloids with the strychnos skeleton (strychnine,
brucine, vomicine, and dimeric forms); 4 in-tar mols all share prec_mz=337.156.

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 184.0   | 6      | 0.559     | 184.040=C₁₁H₆NO₂⁻ and 184.064=C₁₁H₈N₂O⁻ (two series); nan=False |
| 170.1   | 6      | 0.337     | 170.048/170.061 — two series; nan=False |
| 197.0   | 4      | 0.137     | 197.036/197.072 — two series; nan=False |
| 172.1   | 4      | 0.171     | nan=False |
| 198.1   | 4      | 0.092     | |
| 224.1   | 4      | 0.090     | |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 262.1   | 6      | 0.063          | 0.005          | in DAG, model weak |
| 275.1   | 5      | 0.111          | 0.036          | |
| 222.1   | 4      | 0.109          | 0.032          | |
| 303.1   | 4      | 0.093          | 0.014          | |

### Root cause

**Cascade depth (hexacyclic ring system) + very limited tar coverage.**

Only 4/33 mols are in the DAG tar; the 29 missing mols reduce statistical power but the cascade
depth problem is independent. Sample spectra at nce=35 show only precursor (337.156) and a single
large fragment (305.129, loss of 32 Da ≈ CH₃OH), confirming the strychnos skeleton is extremely
fragmentation-resistant. The 184.0 cat A fragments — C₁₁H₆NO₂⁻ (184.040) and C₁₁H₈N₂O⁻
(184.064) from two different mol series — require ring-opening across 4+ bonds in the bridged
hexacyclic framework. No contamination (all nan=False). Category B model weakness is present
but minor (intensity 0.06-0.11).

---

## Perylenequinones (n=24, DAG miss rate=51.7%)

**3 mols; 2 in tar.** Pentacyclic peri-fused aromatic bisquinones with hydroxyl and
alkyl/acyl substituents; includes hypericin-type compounds with prec_mz 351–789 Da.

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 271.1   | 14     | 0.774     | 271.061=C₁₅H₁₁O₅⁻; 2 mols, nan=False; very high intensity |
| 183.0   | 10     | 0.757     | 183.044/183.045; 1 mol, nan=False |
| 233.1   | 12     | 0.499     | 233.060/233.061; 1 mol, nan=False |
| 155.0   | 10     | 0.557     | nan=False |
| 213.0   | 12     | 0.425     | nan=False |
| 232.1   | 10     | 0.276     | |
| 255.0   | 12     | 0.357     | |
| 272.1   | 12     | 0.135     | |
| 234.1   | 12     | 0.139     | |
| 59.0    | 10     | 0.062     | minor |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 256.0   | 12     | 0.594          | 0.001          | 256.034/256.037 — two series; very high intensity |
| 227.0   | 12     | 0.508          | 0.000          | 227.034/227.051 |
| 204.1   | 8      | 0.472          | 0.000          | in DAG, model predicts 0 |
| 228.0   | 12     | 0.424          | 0.007          | |
| 199.0   | 8      | 0.208          | 0.017          | |

### Root cause

**Cascade depth dominant; model weakness secondary.**

The sample spectrum (mol 25008, prec=789.219, nce=40) shows the intact precursor as base peak
with ¹³C isotopes (790.222, 791.225) dominating — even at nce=40 the massive fused aromatic
quinone resists fragmentation. The first accessible fragments (459.109, 474.132) are large
half-molecule ions. Small fragments at 271.061 (C₁₅H₁₁O₅⁻, n=14, 0.774) and 183.044 (n=10,
0.757) require ≥4 bond cuts through the five-ring pericyclic system and are absent from the
BFS DAG. No contamination (nan=0 for all). Category B: 256.0 (0.594) and 227.0 (0.508) are
in the DAG but model predicts ≈0 — model weakness for the perylenequinone ring system.

---

## Phenanthrenes and derivatives (n=68, DAG miss rate=50.9%)

**52 mols; 31 in tar.** Structurally diverse subclass: phenanthrene sulfonates and carboxylates
(mol 49005, prec=257.027 = C₁₄H₉SO₃⁻), catechol-diterpene-phenanthrenes (mols 2349–2352, large
polycyclic), and N-acetyl phenanthryl amines; prec_mz 221–479 Da.

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 193.1   | 11     | 0.625     | 193.066=C₁₄H₉O⁻ from SO₂ loss (sulfonated phenanthrenes, nan=False) |
| 160.0   | 8      | 0.551     | 160.016 — two clusters; genuine aromatic fragment |
| 97.0    | 8      | 0.589     | possible nce=nan phosphate + genuine diterpene fragment |
| 275.1   | 10     | 0.348     | from complex phenanthrene diterpenes |
| 289.1   | 9      | 0.257     | |
| 303.1   | 8      | 0.133     | |
| 147.0   | 8      | 0.082     | |
| 261.1   | 7      | 0.218     | |
| 159.0   | 10     | 0.148     | |
| 197.1   | 6      | 0.104     | |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 200.1   | 13     | 0.608          | 0.001          | 200.011=C₁₁H₄O₄⁻; large model failure |
| 253.1   | 13     | 0.478          | 0.004          | in DAG, model ≈0 |
| 361.2   | 16     | 0.393          | 0.013          | |
| 265.1   | 9      | 0.348          | 0.011          | |
| 347.2   | 12     | 0.394          | 0.014          | |
| 225.1   | 16     | 0.167          | 0.015          | |

### Root cause

**Rearrangement (SO₂ loss) dominant for sulfonated mols; model weakness for cat B; cascade
depth for complex diterpene-phenanthrenes.**

For phenanthrene sulfonates (mol 49005, prec=257.027=C₁₄H₉SO₃⁻): nce=90 MS2 shows 193.066 as
major fragment (50.7% base). The exact mass C₁₄H₉O⁻ = 193.065 — phenanthryl-oxide anion formed
by SO₂ loss rearrangement from the phenanthrene sulfonate. BFS cannot produce this (sulfonate
→ hydroxyphenanthrene requires S-O bond fragmentation + oxygen retention). Category B dominates
for the broader subclass: 200.1 (C₁₁H₄O₄⁻, n=13, 0.608) and 253.1 (n=13, 0.478) are BFS-
reachable but model predicts 0.001/0.004. Category A fragments 275.1/289.1/303.1 from catechol-
diterpene-phenanthrenes (large polycyclics, prec up to 479 Da) require cascade depth >3 cuts.

---

## Pyranoflavonoids (n=40, DAG miss rate=50.4%)

**9 mols; 6 in tar.** Flavonoids with a pyran ring fused to the chromone: furanocoumarin-type
attachment of a 2,2-dimethylchromene unit to flavone/flavanone; mols include mammeisin, cycloart-
type pyranoflavones; prec_mz 303–455 Da.

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 97.0    | 9      | 0.655     | 96.959=H₂PO₄⁻; nan=9/9 (pure nce=nan contamination) |
| 62.0    | 9      | 0.139     | 61.987=NO₃⁻; nan=9/9 |
| 80.0    | 8      | 0.187     | 79.957=SO₃⁻•; nan=8/8 |
| 59.0    | 9      | 0.211     | 59.012=CH₃COO⁻; nan=9/9 |
| 103.1   | 8      | 0.358     | nan=7/8; mostly contamination |
| 143.1   | 7      | 0.091     | minor |
| 115.9   | 6      | 0.115     | .9-bin background; nan=True |
| 99.9    | 6      | 0.445     | .9-bin background; nan=True |
| 105.1   | 6      | 0.193     | |
| 129.1   | 6      | 0.206     | |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 297.1   | 9      | 0.750          | 0.015          | in DAG at high intensity; model severely underpredicts |
| 147.0   | 8      | 0.470          | 0.022          | in DAG, model underpredicts |
| 157.1   | 6      | 0.072          | 0.016          | |
| 172.1   | 5      | 0.104          | 0.032          | |
| 130.0   | 6      | 0.103          | 0.003          | |

### Root cause

**Data quality dominant (cat A); model weakness for genuine fragments (cat B).**

All top cat A peaks have nan=True for every single contributing spectrum (9/9 or 8/8), reproducing
exactly the nce=nan mobile phase contamination profile: H₂PO₄⁻ (97.0), NO₃⁻ (62.0), SO₃⁻•
(80.0), CH₃COO⁻ (59.0), .9-bin series (115.9/99.9). 18/46 spectra from in-tar mols are nce=nan.
Category B reveals the genuine problem: 297.1 (n=9, 0.750 intensity) and 147.0 (n=8, 0.470) are
real BFS-reachable pyranoflavonoid fragments that the model assigns ≈0 probability.

---

## Oxocins (n=50, DAG miss rate=48.9%)

**20 mols; 13 in tar.** Eight-membered ring oxygen heterocycles: terpenoid oxocin lactones,
sesqui- and diterpene-oxocin natural products (including eunicellin-type diterpenoids and
guaianolide-derived oxocins); prec_mz 243–463 Da.

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 135.0   | 9      | 0.769     | 134.89–134.92 exact; n=9 from analysis, nan varies; very high intensity |
| 181.1   | 12     | 0.606     | 181.051=C₉H₉O₄⁻; at low nce=2 in sample spectrum (mol 9918) |
| 151.0   | 12     | 0.338     | nan varies |
| 123.0   | 10     | 0.511     | nan=9/25 in probe |
| 137.0   | 10     | 0.443     | nan=9/26 in probe |
| 107.1   | 8      | 0.405     | |
| 199.1   | 9      | 0.092     | |
| 171.1   | 10     | 0.063     | |
| 163.0   | 12     | 0.205     | |
| 377.0   | 12     | 0.064     | |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 333.2   | 9      | 0.599          | 0.023          | high-intensity genuine terpenoid fragment; model weak |
| 257.1   | 10     | 0.412          | 0.002          | in DAG, model ≈0 |
| 173.1   | 8      | 0.307          | 0.002          | |
| 315.2   | 13     | 0.252          | 0.024          | |
| 229.2   | 8      | 0.165          | 0.000          | |

### Root cause

**Cascade depth + rearrangement (8-membered ring); model weakness secondary.**

The 8-membered ring in terpenoid oxocins opens via ring contraction/transannular rearrangement,
producing fragments that BFS cannot reach. Fragment 181.051 (C₉H₉O₄⁻) appears at only nce=2 in
sample mol 9918 (prec=463.197) — it's a low-energy product formed by the ring-opening
rearrangement of the oxocin/lactone system, not a simple bond cut. For the large sesquiterpenoid
oxocins, 377.0 (n=12) appearing as the second most common cat A peak reflects the difficulty of
producing the large-ring fragment even at depth 1. Category B: 333.2 (0.599) and 257.1 (0.412)
are BFS-reachable but model assigns near-zero probability. Data quality (9/32 nce=nan spectra)
contributes modestly to some cat A peaks (123.0/137.0).

---

## Phenol esters (n=72, DAG miss rate=48.6%)

**50 mols; 46 in tar.** Esters of phenols with aliphatic or aromatic acids: methacrylic acid
phenol esters, acetylated catechols, thiophene-phenol amides (mol 3405), and larger diaryl
esters (up to prec=441 Da); 6/56 FT spectra are nce=nan.

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 97.0    | 6      | 0.642     | nan=6/6 — 100% nce=nan contamination (H₂PO₄⁻) |
| 80.0    | 5      | 0.294     | nan=5/5 — SO₃⁻• contamination |
| 99.9    | 4      | 0.206     | nan=4/4 — .9-bin background |
| 115.9   | 4      | 0.084     | nan=4/4 — .9-bin background |
| 62.0    | 3      | 0.144     | nan=3/3 — NO₃⁻ contamination |
| 79.0    | 3      | 0.071     | nan=3/3 — PO₃⁻ contamination |
| 230.9   | 3      | 0.239     | nan=5/13 in probe; .9-bin possible halide |
| 195.0   | 5      | 0.144     | nan=6/24 in probe; mixed |
| 166.1   | 4      | 0.094     | minor |
| 194.1   | 3      | 0.076     | minor |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 118.0   | 5      | 0.853          | 0.014          | very high intensity; in DAG, model ≈0 |
| 59.0    | 4      | 0.532          | 0.018          | acetate or ester fragment; in DAG, model weak |
| 135.0   | 8      | 0.121          | 0.028          | phenolate-related fragment |
| 93.0    | 6      | 0.101          | 0.019          | 93.034=phenolate [C₆H₅O]⁻; in DAG, model weak |
| 123.0   | 4      | 0.239          | 0.006          | |

### Root cause

**Data quality dominant (cat A); model weakness secondary (cat B).**

Every cat A peak with n≥3 has nan=True for 100% of contributing spectra, reproducing the
standard nce=nan mobile phase contaminant fingerprint (H₂PO₄⁻/SO₃⁻•/NO₃⁻/PO₃⁻/.9-bin series).
Category B reveals genuine failures: 118.0 (0.853) and 59.0 (0.532) are BFS-reachable but model
assigns near-zero probability. Phenolate anion (93.034=C₆H₅O⁻) appears in cat B (n=6, in DAG,
model 0.019) confirming direct-anion fragmentation from ester hydrolysis IS accessible to BFS
but underweighted. The 230.9 cat A bin (.9 fractional mass, n=3) may involve a halide-retaining
fragment from a specific chlorinated phenol ester.

---

## Anthraquinones (n=818, DAG miss rate=48.5%)

**214 mols; 173 in tar.** The second-largest subclass (818 analysis spectra): sulfonated,
aminated, and hydroxy anthraquinone dyes/pigments/natural products; derivatives include vat
dyes, aminoanthraquinones, and anthraquinone carboxylates (mols with prec up to 772 Da).

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 182.0   | 81     | 0.292     | multiple aromatic fragment series; same pattern as anthracenes |
| 210.0   | 69     | 0.230     | anthraquinone ring fragment; nan=27/81 (mostly genuine) |
| 154.0   | 69     | 0.167     | |
| 183.0   | 68     | 0.378     | ALSO appears in cat B (see below) — depends on mol |
| 169.0   | 66     | 0.423     | identical bin to anthracenes cat A (cascade depth) |
| 141.0   | 66     | 0.353     | identical bin to anthracenes cat A (cascade depth) |
| 253.1   | 63     | 0.414     | CO₂ loss series from anthraquinone carboxylate: [M-H-2CO₂]⁻ |
| 195.0   | 58     | 0.463     | ALSO in cat B; two different populations |
| 211.0   | 57     | 0.467     | |
| 171.0   | 57     | 0.249     | |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 195.0   | 53     | 0.400          | 0.004          | in DAG, model ≈0 (different mols from cat A 195.0) |
| 183.0   | 45     | 0.323          | 0.010          | in DAG, model ≈0 (different mols from cat A 183.0) |
| 113.0   | 44     | 0.286          | 0.002          | |
| 139.1   | 43     | 0.135          | 0.002          | |
| 224.0   | 43     | 0.192          | 0.010          | |
| 227.0   | 40     | 0.377          | 0.010          | |
| 226.0   | 39     | 0.348          | 0.014          | |
| 101.0   | 36     | 0.157          | 0.005          | |

### Root cause

**Cascade depth dominant (same as anthracenes); model weakness secondary.**

Bins 169.0 and 141.0 match the anthracene cat A pattern exactly, confirming the same cascade
mechanism: the three-ring anthraquinone core (plus carbonyl groups) requires ≥4 sequential bond
cuts to produce small aromatic ions (C₁₂H₁O₂⁻, C₉H₅O⁻, etc.). Sample mol 11232 (prec=443.098,
nce=14) shows successive CO₂ losses: 443.099→399.073 (−CO₂)→355.083 (−2CO₂), with nothing
below 326 at mild CE. 253.1 = [anthraquinone carboxylate − 2CO₂]⁻ (confirmed by the loss
series). Bins 182.0/183.0/195.0 appear in both cat A and cat B because the BFS can reach them
from some mols but not others, depending on substitution pattern. Category B: 195.0 (n=53, 0.40)
and 227.0 (n=40, 0.377) are in DAG at high intensity but model predicts ≈0.

---

## Oxepanes (n=101, DAG miss rate=47.6%)

**51 mols; 34 in tar.** Seven-membered ring oxygen heterocycles in terpenoid natural products:
monoterpene oxepane epoxides, diterpene oxepane-fused macrolides, and sesquiterpenoid lactones
with an oxepane ring embedded in complex polycyclic skeletons; prec_mz 161–837 Da.

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 247.1   | 18     | 0.784     | 247.097=[M-2H]²⁻ from mol 11626 (C₂₈H₃₂O₈, M=496.209); 17/18 spectra from 1 mol; BFS cannot generate doubly-charged ions |
| 137.0   | 18     | 0.281     | 137.024/137.023 — two series; nan=2/55 |
| 151.0   | 22     | 0.233     | 151.039/151.040; nan=2/38 |
| 203.1   | 16     | 0.096     | minor |
| 95.0    | 15     | 0.464     | nan=4/79 → mostly genuine aromatic fragment |
| 187.1   | 15     | 0.095     | minor |
| 173.1   | 15     | 0.136     | nan=2/43 → genuine |
| 175.0   | 14     | 0.201     | |
| 159.1   | 19     | 0.088     | |
| 97.0    | 14     | 0.193     | nan=? possible contamination + genuine mix |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 136.1   | 14     | 0.525          | 0.015          | 136.016; nan=1/33 → genuine; in DAG, model ≈0 |
| 117.0   | 17     | 0.395          | 0.015          | 116.986/117.019; in DAG, model ≈0 |
| 111.0   | 15     | 0.207          | 0.006          | |
| 57.0    | 25     | 0.127          | 0.023          | |
| 83.0    | 20     | 0.126          | 0.023          | |

### Root cause

**Doubly-charged [M-2H]²⁻ dominant (large glycoside); cascade depth + model weakness secondary.**

The highest-intensity cat A peak at 247.097 is not a fragment but the [M-2H]²⁻ doubly-charged
precursor ion from mol 11626 (C₂₈H₃₂O₈, MW=496.209, [M-H]⁻=495.201; [M-2H]²⁻ = 247.097 Da).
17 of 18 contributing spectra come from this single mol. BFS generates only singly-charged
fragment ions and cannot produce doubly-charged species. The remaining cat A peaks (137.0/95.0/
151.0) are genuine terpenoid ring fragments requiring 3+ sequential cuts after 7-membered lactone
ring-opening, inaccessible to BFS. Only 4/102 spectra are nce=nan (minimal contamination).
Category B: 136.1 (0.525, nan=1/33 → genuine) and 117.0 (0.395) are in DAG but model predicts ≈0.

---

## Pyranocoumarins (n=92, DAG miss rate=47.4%)

**36 mols; 28 in tar.** 2H-chromene or 2H-pyran ring fused to a coumarin core: visnagin,
khellin, mammisine, and related angular/linear pyranocoumarin natural products from Apiaceae;
prec_mz 227–427 Da; 40% of in-tar spectra are nce=nan (33/82).

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 111.0   | 27     | 0.164     | nan=28/34 → mostly nce=nan contamination |
| 62.0    | 23     | 0.200     | nan=33/42 → near-total contamination (NO₃⁻) |
| 80.0    | 20     | 0.369     | nan=33/43 → near-total contamination (SO₃⁻•) |
| 95.0    | 20     | 0.262     | nan=20/43 |
| 261.1   | 20     | 0.367     | nan=15/38 → 23 non-nan; GENUINE fragment absent from DAG |
| 97.0    | 19     | 0.705     | nan=33/53 → near-total contamination (H₂PO₄⁻) |
| 115.1   | 19     | 0.260     | nan=28/53 → mostly contamination |
| 209.1   | 18     | 0.283     | nan=15/? |
| 79.0    | 16     | 0.113     | nan=33/? → contamination |
| 81.0    | 16     | 0.319     | mixed |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 265.1   | 24     | 0.145          | 0.018          | genuine pyranocoumarin fragment; model underpredicts |
| 289.1   | 22     | 0.249          | 0.005          | |
| 291.1   | 20     | 0.243          | 0.016          | |
| 305.1   | 18     | 0.315          | 0.009          | in DAG, model ≈0 |
| 277.1   | 17     | 0.238          | 0.013          | |
| 251.1   | 17     | 0.135          | 0.011          | |
| 275.1   | 17     | 0.206          | 0.012          | |

### Root cause

**Data quality dominant (nce=nan contamination); model weakness secondary for genuine fragments.**

40% of spectra are nce=nan. The primary cat A peaks are mobile phase contaminants: NO₃⁻ (62.0,
nan=33/42), SO₃⁻• (80.0, nan=33/43), H₂PO₄⁻ (97.0, nan=33/53), and background .9-bin series
(115.1, 111.0). Peak 261.1 (n=20, 0.367, nan=15/38) has 23 non-nan contributing spectra,
suggesting a genuine pyranocoumarin fragment absent from the BFS DAG — possibly a furan-ring-
opening product from the pyranocoumarin oxygen bridge. Category B has multiple genuine
pyranocoumarin fragments in the DAG at 0.14-0.32 intensity that the model systematically
underpredicts: 265.1/289.1/305.1/291.1 all show model predictions of 0.005-0.018.

---

## P-quinodimethanes (n=30, DAG miss rate=46.7%)

**5 mols; 4 in tar.** Para-methylene-cyclohexadiene derivatives including mellitic acid (mol
50616, benzenehexacarboxylic acid), hydroxybenzenedicarboxylate quinodimethanes, and an
oxazolone-styryl conjugate (mol 2365); prec_mz 211–347 Da.

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 107.0   | 15     | 0.478     | 107.050=C₇H₇O⁻; nan=3/26 → mostly genuine |
| 151.0   | 9      | 0.579     | 151.040=C₈H₇O₃⁻; nan=0! — entirely genuine |
| 123.1   | 8      | 0.560     | 123.008=C₆H₃O₃⁻ and 122.678 (two series); nan=0 for main series |
| 91.1    | 8      | 0.272     | 91.018=C₆H₃O⁻; nan=2/19 |
| 109.1   | 8      | 0.207     | |
| 125.0   | 7      | 0.101     | |
| 97.0    | 7      | 0.099     | |
| 93.0    | 6      | 0.229     | |
| 135.0   | 8      | 0.234     | |
| 65.0    | 9      | 0.198     | |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity | Note |
|---------|--------|----------------|----------------|------|
| 111.0   | 10     | 0.667          | 0.003          | 111.008=C₅H₃O₃⁻; nan=2 → genuine; model ≈0 |
| 81.0    | 8      | 0.576          | 0.000          | 81.034=C₅H₅O⁻; nan=0! — genuine; model 0 |
| 99.0    | 7      | 0.536          | 0.010          | 99.008=C₄H₃O₃⁻; nan=0 — genuine |
| 71.0    | 11     | 0.451          | 0.001          | 71.013=C₃H₃O₂⁻; nan=0 — genuine |
| 69.0    | 13     | 0.312          | 0.016          | |
| 57.0    | 9      | 0.295          | 0.002          | |
| 96.0    | 8      | 0.284          | 0.017          | |
| 67.0    | 7      | 0.392          | 0.021          | 67.019 |

### Root cause

**Model weakness dominant; cascade depth secondary.**

Category B is the primary failure: 111.0 (C₅H₃O₃⁻, n=10, 0.667 intensity), 81.0 (C₅H₅O⁻,
n=8, 0.576, nan=0!), 71.0 (C₃H₃O₂⁻, n=11, 0.451, nan=0), and 99.0 (C₄H₃O₃⁻, n=7, 0.536,
nan=0) are all BFS-reachable and contamination-free yet the model predicts 0.000-0.016. These
are deprotonated small oxygen-containing aromatics from the polyfunctionalized benzene core that
the model systematically assigns near-zero probability. Category A: 151.0 (nan=0, intensity
0.579) and 123.1 (nan=0, intensity 0.560) are absent from the BFS DAG despite clean spectra.
For mellitic acid (mol 50616, benzenehexacarboxylic acid), sequential CO₂ losses require
>3 bond cuts to reach 151.0 and 123.0 — cascade depth locks those fragments out.

## Summary

| Subclass | n | miss% | Root cause summary |
|----------|---|-------|--------------------|
| Cholestane steroids | 32 | 100.0% | **Artifact** (14/17 mols not in tar) + **direct anion** (HSO₄⁻, SO₃⁻• from cholesteryl sulfate 6778); 466.3 = ¹³C precursor isotope in MS2 |
| Flavonoids | 24 | 84.7% | **Data quality** (sulfate/phosphate mobile-phase contamination in n=nan-CE flavone spectra; 223.0 > precursor 221.06 → co-isolation) + **direct anion** (Br-retaining .9 bins from 6-bromoflavone 50271) |
| Pyrenes | 65 | 82.8% | **Rearrangement** (SO₃/SO₂ cascade: 233.1, 217.1, 313.0 from polysulfonate pyrenes; CO loss 189.1 from hydroxypyrene) + **data quality** (¹³C isotope peaks 457.9, 234.1, 190.1 appearing as missing fragments) |
| Pyrimidodiazepines | 30 | 73.3% | **Data quality** (all in-tar spectra from mol 21531 are nce=nan, dominated by CH₃COO⁻/H₂PO₄⁻/NO₃⁻ mobile phase contaminants; above-precursor co-isolation peak at 395.2) |
| Sulfinylbenzimidazoles | 26 | 71.4% | **Rearrangement** (SO₂ loss from aminosulfonylbenzimidazole → 132.056=[C₇H₆N₃]⁻; SO₃ loss from amino-2-sulfonylbenzimidazole → 132.032; all at 132.0/132.1 bins) |
| 9,9p-dihydroxyaryltetralin lignans | 58 | 68.4% | **Rearrangement** (retro-Diels-Alder of tetralin ring + radical C₁₃ fragment at 196.052=C₁₃H₈O₂⁻•; CO-loss series 253.1→225.1→197.1, Δ=28.0; all 5 mols affected) + **model weakness** (cat B: guaiacyl fragments 159.0, 181.1, 191.1 in DAG but underpredicted) |
| Pyridine carboxaldehydes | 44 | 66.7% | **Direct anion** (pyridoxal-5-phosphate: PO₃⁻=78.96 base peak, H₂PO₄⁻=96.97; phosphate head group fragments dominate) + **data quality** (247.0 = ¹³C isotope of [M-H]⁻ leaking into MS2) |
| Aporphines | 75 | 64.3% | **Model weakness** dominant (265.1/250.1 = C₁₆H₉O₄⁻ in DAG at 0.701 intensity, 53-67 spectra, model predicts 0.007/0.006; 163.1 at 0.602 also) + **cascade depth** (224.0 absent from DAG across 29 spectra) |
| Tropane alkaloids | 22 | 63.7% | **Data quality** (mol 21573 nce=nan: stearate 283.264 base peak; all .9-bin cat A + phosphate/nitrate/sulfate from contaminated spectra) + **model weakness** (cat B: 365.2/508.3 at ~0.53 intensity, model ≈0) |
| Furanocoumarins | 120 | 59.0% | **Data quality** overwhelmingly dominant (nce=nan mobile-phase contamination: H₂PO₄⁻/NO₃⁻/SO₃⁻•/SO₂⁻•/stearate across 26-28 diverse mols; 97.0 n=90, 62.0 n=78, 283.3 n=28) + **model weakness** for genuine fragments in cat B |
| Isoindoles | 73 | 59.0% | **Model weakness** dominant (108.0 = base peak in DAG intensity=1.000, model predicts 0.001; 166.1 at 0.879, model 0.000; 69.0 at 0.733, model 0.003) + **cascade depth** (282.1/254.1 absent from DAG for large prec=549.16 mol) |
| Lignan glycosides | 379 | 58.3% | **Data quality** dominant (nce=nan; 136.0 n=242, 151.0 n=271 across 34 mols with nan=True) + **rearrangement** (196.1/197.1 = C₁₃ aryltetralin radical aglycone, same motif as aryltetralin lignans) + **model weakness** (cat B: 145.0 n=46, 153.1 n=31) |
| Delta valerolactones | 35 | 57.0% | **Data quality** dominant (nce=nan: 62.0=NO₃⁻, 97.0=H₂PO₄⁻, 80.0=SO₃⁻•, 64.0=SO₂⁻•, 115.9=.9-bin; same contaminant profile as furanocoumarins) + **model weakness** (191.1 at 0.686 in DAG, model 0.001) |
| Anthracenes | 121 | 55.8% | **Cascade depth** (169.0/141.0/153.0/197.0 are genuine aromatic fragments from 10-12 diverse mols with nan=False; tricyclic aromatic ring requires ≥4 bond cuts to produce C₉-C₁₃ fragments) + **model weakness** (cat B: 343.0 n=15 at 0.379, model 0.007) |
| Morphinans | 87 | 54.8% | **Model weakness** dominant (225.1 n=41 intensity 0.562 in DAG model 0.014; 283.1 n=16 at 0.723 model 0.006; no nce=nan contamination) + **cascade depth** (185.1/197.1 absent from DAG → pentacyclic ring cascade ≥4 cuts) |
| Coumestans | 24 | 54.8% | **Model weakness** dominant (279.0 n=19 intensity 0.738 in DAG model 0.005; 223.0 at 0.527 model 0.022; 266.0 at 0.492 model 0.015; no contamination) + **rearrangement** (195.0 absent from DAG → coumestan ring-opening) |
| Strychnos alkaloids | 22 | 53.7% | **Cascade depth** (hexacyclic strychnos ring; 184.0=C₁₁H₆NO₂⁻/C₁₁H₈N₂O⁻ absent from DAG; sample shows only precursor + 1 fragment even at nce=35) + **limited tar coverage** (4/33 mols in tar) |
| Perylenequinones | 24 | 51.7% | **Cascade depth** (pentacyclic pericyclic quinone; precursor dominates even at nce=40; 271.1 n=14 intensity 0.774 absent from DAG) + **model weakness** (cat B: 256.0 at 0.594, 227.0 at 0.508, 204.1 at 0.472 — model predicts ≈0) |
| Phenanthrenes and derivatives | 68 | 50.9% | **Rearrangement** (SO₂ loss from sulfonated phenanthrenes → C₁₄H₉O⁻=193.066; same mechanism as pyrene sulfonates) + **model weakness** (200.1 n=13 at 0.608 in DAG model 0.001; 253.1 at 0.478) + **cascade depth** (catechol-diterpene mols) |
| Pyranoflavonoids | 40 | 50.4% | **Data quality** dominant (nce=nan: 97.0/62.0/80.0/59.0/115.9/99.9 all 100% nce=nan contamination; 18/46 spectra nan) + **model weakness** (297.1 n=9 at 0.750 in DAG, model 0.015; 147.0 at 0.470) |
| Oxocins | 50 | 48.9% | **Cascade depth + rearrangement** (8-membered ring opening; 181.1=C₉H₉O₄⁻ at low nce=2 confirming ring-rearrangement product; 135.0 intensity 0.769 absent from DAG) + **model weakness** (333.2 at 0.599, 257.1 at 0.412) |
| Phenol esters | 72 | 48.6% | **Data quality** dominant (cat A: ALL peaks 100% nce=nan; H₂PO₄⁻/SO₃⁻•/NO₃⁻/PO₃⁻/.9-bins) + **model weakness** (118.0 at 0.853 and phenolate 93.034 in DAG, model ≈0) |
| Anthraquinones | 818 | 48.5% | **Cascade depth** identical to anthracenes (169.0/141.0 in both; same ring system; CO₂ loss cascade 443→399→355 in sample) + **model weakness** (cat B: 195.0 n=53 at 0.400, 227.0 n=40 at 0.377 in DAG, model 0.004-0.010) |
| Oxepanes | 101 | 47.6% | **Doubly-charged [M-2H]²⁻** (mol 11626, C₂₈H₃₂O₈, M=496; 247.097=17/18 spectra; BFS generates only singly-charged ions) + **cascade depth** (95.0/137.0/151.0) + **model weakness** (136.1 at 0.525→0.015) |
| Pyranocoumarins | 92 | 47.4% | **Data quality** dominant (40% nce=nan spectra; 62.0/80.0/97.0 pure contamination; nan=33/42-53) + **model weakness** (289.1/305.1/291.1/265.1 at 0.14-0.32 in DAG, model 0.005-0.017) + genuine fragment 261.1 (23 non-nan spectra) |
| P-quinodimethanes | 30 | 46.7% | **Model weakness** dominant (cat B: 111.0=0.667/81.0=0.576/99.0=0.535/71.0=0.451 all nan=0, in DAG, model predicts 0.000-0.016) + **cascade depth** (151.0/123.1 nan=0 absent from DAG; mellitic acid needs >3 CO₂ losses) |
