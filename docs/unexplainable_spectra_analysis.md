# Unexplainable-Spectra Analysis (DAG Oracle Ceiling)

A workflow for finding and diagnosing NIST spectra that the fragmentation DAG **cannot explain**,
grouped by adduct, molecule, and ClassyFire class. "Unexplainable" is measured by the **oracle
cosine ceiling** — the best cosine any model could reach given only the DAG's fragment coverage
(true intensities placed on DAG-achievable peaks, zero elsewhere). A low ceiling means the DAG
itself is the bottleneck, independent of the trained model.

All scripts live in `scripts/eval/` and read the per-spectrum oracle CSV produced by
`dag_oracle_scores.py`. Default paths point at the NIST20 `default` scoring run in
`data/eval/oracle_cos_nist20_default/`.

## Concepts

- **Oracle ceiling** (`cos_hun_*_opt`): DAG coverage ceiling per spectrum. This analysis thresholds
  on `cos_hun_inst_tol_opt` — the **per-instrument** tolerance variant (FT 10 ppm / floor 200 Da,
  QTOF 20 ppm / floor 750 Da), i.e. the tolerance the model is actually evaluated at.
- **Unexplainable**: oracle ceiling `< 0.5` (configurable).
- **Zero-cos tail**: ceiling `== 0` — the DAG matches *no* peak. All-or-nothing coverage failure.
- **Filter tier**: the source CSV is the `1_above_prec` tier (peaks above the precursor dropped,
  no UWPR denoise); `n_uwpr_removed == 0` throughout confirms this. Any re-scoring reproduces it.

## Scope used for the headline numbers

Instruments `{FT, QTOF}` and adducts `{[M+H]+, [M-H]-, [M+Na]+, [M+Cl]-}`. This drops the noisy
`Other`/`IT`/`QQQ` instruments and the `[M+K]+` adduct (whose fragment masses the DAG does not
model), which otherwise dominate a misleading tail. Scope = 552,824 spectra / 25,938 molecules.

## Pipeline

```
dag_oracle_scores.py            # (upstream) per-spectrum oracle ceiling CSV
        │
        ▼
unexplainable_analysis.py       # summarize by adduct / class; dump unexplainable + zero-cos tables
        │
        ├── zero_cos_recoverability.py   # re-score zero-cos tail at 0.05 Da: recoverable vs broken
        │
        └── export_zero_cos.py           # MGF + metadata export for external formula decomposition
```

### 1. `unexplainable_analysis.py`

Restricts to the scope, attaches structure (`num_atoms`, formula, SMILES) and ClassyFire class,
then writes tables and two figures.

```bash
python scripts/eval/unexplainable_analysis.py \
    --out_dp data/eval/oracle_cos_nist20_default \
    --inst_types FT QTOF \
    --prec_types "[M+H]+" "[M-H]-" "[M+Na]+" "[M+Cl]-" \
    --cos_col cos_hun_inst_tol_opt --thresh 0.5
```

Outputs:
- `unexpl_spectra.csv`, `unexpl_mols.csv`, `unexpl_by_classyfire_klass.csv`
- `zero_cos_spectra.csv`
- `unexplainable_overview.png` (ceiling histogram; % unexplainable by adduct×instrument and class)
- `zero_cos_diagnostic.png` (zero-cos by adduct×instrument; sparsity; by class)

### 2. `zero_cos_recoverability.py`

Splits the zero-cos tail into **recoverable** (borderline miscalibration) vs **broken** (wrong
annotation / gross mass error / noise) by re-scoring only the affected molecules at a fixed
**0.05 Da** tolerance. 0.05 Da is chosen because it is looser than *both* instrument windows — a
fixed 0.01 Da probe is invalid for QTOF (tighter than its own 20 ppm, so it can never recover a
QTOF miss). Re-scoring reuses `dag_oracle_scores.score_dag` and **asserts** it reproduces the
source `1_above_prec` tier before trusting the result.

```bash
python scripts/eval/zero_cos_recoverability.py \
    --frag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags \
    --out_dp data/eval/oracle_cos_nist20_default
```

Outputs: `zero_cos_recoverability.csv` (per-spectrum `recoverable_0p05da`, `n_ach_0p05da`) and
`zero_cos_recoverability.png` (recoverable/broken split by instrument; sparsity; broken by class).

> Requires the extra tolerance columns added to `dag_oracle_scores.py`
> (`n_achievable_peaks_0p05da`, `cos_hun_0p05da_opt`). Older CSVs lack them, so this script
> re-scores rather than reading them from the CSV.

### 3. `export_zero_cos.py`

Exports the zero-cos spectra for external formula decomposition (e.g. MSBuddy), attaching two
independent triage signals:

- **Cross-CE control** (`best_sibling_cos`, `n_siblings`, `all_ce_zero`): the DAG is
  **CE-independent** — one DAG serves all collision energies of a given (molecule, adduct). So if a
  sibling spectrum at another CE is well explained, coverage is proven fine and the cos-0 is a
  *per-spectrum acquisition artifact*, not a coverage gap. `all_ce_zero` marks the systematic
  residue where every CE fails.
- **0.05 Da recoverability** (`recoverable_0p05da`), merged from step 2 if present.

```bash
python scripts/eval/export_zero_cos.py --out_dp data/eval/oracle_cos_nist20_default
```

Outputs:
- `zero_cos.mgf` — one `BEGIN IONS` block per spectrum (`TITLE`/`SCANS`=spec_id, `PEPMASS`,
  signed `CHARGE`, `ADDUCT`, raw peaks).
- `zero_cos_export_meta.csv` — per-spectrum metadata (triage flags, `formula`, `exact_mw`,
  `num_atoms`, `charge`, ClassyFire, SMILES), sorted with the systematic residue first.

## Key findings (NIST20, scope above)

- **Overall ceiling is high**: median 1.000, mean 0.950 (inst_tol). Unexplainable (`< 0.5`) is
  **3.43%** (18,960 spectra / 5,399 molecules). Molecule size is not the driver — 99.9% of
  unexplainable spectra sit inside the training heavy-atom range [4, 104].
- **Adduct**: `[M+Na]+` is the hardest of the four (6.7% FT); `[M+H]+`/`[M-H]-` are cleanest
  (2–4%); `[M+Cl]-` is near-fully explainable.
- **ClassyFire**: unexplainability concentrates in **fused/rigid polycyclic scaffolds** needing
  multi-bond ring cleavage — morphinans (29%), anthracenes (19%), phenanthrenes, naphthopyrans,
  coumarins/benzopyrans, prenol-lipid terpenoids, steroids. Flat/acyclic chemistry (organic acids,
  amino acids, azoles, pyridines) is near-ceiling.
- **Zero-cos tail (647 spectra / 340 molecules)** is a diffuse, per-spectrum **data-quality**
  problem, not localized chemistry (mean 1.9 spectra/mol; 56% of molecules contribute a single
  spectrum). Two independent controls agree:
  - **Cross-CE**: 80% have a sibling with cos ≥ 0.70 (76% ≥ 0.90) at another CE → coverage proven
    fine → acquisition artifact. Only **37 (6%) are all-CE-zero** (systematic).
  - **0.05 Da re-score**: 75% recoverable (FT 85%, QTOF 55%), only **25% (162) genuinely broken**.
  - The genuinely-broken residue is small polar classes (carboxylic acids, steroids, organooxygen,
    fatty acyls) with off-mass or noise peaks — a *filtering* target, not a modeling one.

## Interpreting an external formula-decomposition (MSBuddy) pass

For a zero-cos peak set decomposed against the precursor's element budget:

- **No valid sub-formula** for the peaks → the m/z are wrong (miscalibration / contaminant) →
  drop the spectrum.
- **Valid formula that is absent from the DAG** → a genuine fragmentation-coverage gap → a
  candidate for a new SMARTS/neutral-loss rule or DAG extension.

The cross-CE control already clears ~80% of the tail as artifacts, so a decomposition pass is only
needed for the `all_ce_zero` + solo (no-sibling) rows — the ~65 spectra a sibling cannot vindicate.
