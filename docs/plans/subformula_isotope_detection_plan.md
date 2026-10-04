# Subformula-Sourced Fragment Formulas for Isotope Detection

> **Status:** LANDED (working tree, uncommitted) &nbsp;·&nbsp; **Updated:** 2026-09-20
>
> Steps 0-5 done. The fragment-composition source in target-side isotope cleanup is now
> pluggable, and the subformula source needs no DAG and no `frag_dp`. Measured head to head on
> 1,500 FT NIST20 spectra it **matches the DAG source** (F1 0.808 vs 0.805 native, 0.876 vs
> 0.878 with group-f; 99.2% of peak decisions identical), so the Step 4 gate passed and the
> config surface shipped. **The feature itself is still off in every template** and no training
> run has used it - `remove_isotope_formula_source: none` by default. Step 6 (soft output) not
> started. Read "Results" before proposing anything here.

## Motivation

Target-side isotope cleanup (`spec_params.remove_isotope_*`) needs a **fragment composition** to
compute the expected M+k/M+0 ratio. Today it gets it from the fragmentation DAG:
`remove_isotope_use_dag_formula` + `remove_isotope_dag_dp` load a per-molecule DAG and hand
`(formula_peak_mzs[:, 0], idx_to_formula)` to the detector
([base_dataset.py:568-609](../../src/fragnnet/dataset/base_dataset.py#L568)).

Two problems follow from that choice:

1. **It is FraGNNet-only.** `frag_dp` sat in all ten config templates purely so this branch could
   be fed for any model. As of 2026-09-21 the per-model templates are replaced by the single
   `config/template.yml`, which declares `frag_dp` for every model type,
   so on any other model the DAG branch can no longer be satisfied at all.
2. **A DAG is a model artifact, not a chemistry bound.** It answers "can my fragmenter enumerate
   this?", while the question being asked is "what composition can this peak be?". The subformula
   filter in `clean_spectra_utils.py` already answers exactly the second question, model-free, and
   says so in `_precursor_ion_budgets`: *"nothing here consults a DAG"*
   ([clean_spectra_utils.py:185-227](../../src/fragnnet/utils/clean_spectra_utils.py#L185)).

Swapping the source makes the cleanup available to every model with no DAG, no `frag_dp`, and no
preprocessing dependency, and it removes the last reason to re-add `frag_dp` to a non-FraGNNet
template.

This plan does **not** propose re-enabling the cleanup. It only proposes making it sourceable
without a DAG. Turning it on again is a separate decision that needs the Step 4 numbers.

## What the DAG is used for today

| Use | Code | Subformula analogue |
|-----|------|---------------------|
| Exact fragment composition at an M+0 peak, for `_expected_mk_ratio` | `_find_dag_formula` → `_fragment_expected_ratio` ([isotope_utils.py:415-465](../../src/fragnnet/utils/isotope_utils.py#L415)) | Candidate set from a mass-window subformula decomposition; point estimate when the candidates agree, interval when they do not (Step 3) |
| `protect_dag_mono`: never flag a peak that is itself a DAG M+0 node | `protect_known_mono` ([isotope_utils.py](../../src/fragnnet/utils/isotope_utils.py)) | Not a drop-in - but **measured the opposite way round from the guess below**: DAG protection destroys recall, subformula protection is the highest-precision arm. See Results |
| `dag_filter`: require the parent M+0 to exist in the DAG (debug-script arm only) | [debug_dag_iso_peak_detection.py](../../scripts/debug/isotopes/debug_dag_iso_peak_detection.py) | Require the parent M+0 to be subformula-feasible — strictly weaker, cheap, safe |

The fallback when no DAG formula is found is the precursor composition mass-scaled by
`neutral_frag / prec_neutral_mass` ([isotope_utils.py:462-465](../../src/fragnnet/utils/isotope_utils.py#L462)).
That fallback is what the subformula source is competing against, and it is known to be wrong for
Cl/Br (blocker 3 in [detect_isotope_peaks_tuning.md](../reference/detect_isotope_peaks_tuning.md)):
halogens are all-or-nothing carriers, not mass-scalable.

## Preliminary measurement (indicative, taken 2026-09-20)

Superseded for coverage by the Results section, which measures the real formula distribution.
Kept for the carbon-count spread, which Results does not repeat.

`_subformula_masses(formula, "[M+H]+")`, plus the fraction of random m/z in [100, prec] that land
within FT's 4 mDa of some candidate:

| Precursor formula | candidate ion masses | enumeration | P(random m/z within 4 mDa) |
|---|---:|---:|---:|
| C9H8O4 | 505 | <10 ms | 0.022 |
| C21H30O2 | 1,636 | <10 ms | 0.048 |
| C27H46O | 2,110 | <10 ms | 0.041 |
| C17H19NO3 | 2,844 | <10 ms | 0.082 |
| C43H66N12O12S2 | 1,348,086 | 0.17 s | 0.501 |

And the spread of carbon count among candidates within 4 mDa of each candidate m/z (ad-hoc
`[frag+H]+`-budget enumeration, no S/Cl/Br, so treat as indicative):

| Precursor formula | nC spread p50 / p90 / max | relative M+1-ratio spread p50 / p90 |
|---|---|---|
| C9H8O4 | 0 / 0 / 0 | 0.00 / 0.00 |
| C27H46O | 0 / 0 / 0 | 0.00 / 0.00 |
| C17H19NO3 | 0 / 3 / 3 | 0.00 / 0.23 |

Read: for ordinary small molecules the subformula candidate set pins the carbon count **uniquely
at the median peak**, so the expected M+1 ratio is a point estimate exactly as with a DAG formula;
N-rich molecules carry a tail where it is not. Both the cost and the discriminating power fall
apart for peptide-sized precursors — the grid there explains half of all m/z, which is why
`protect_dag_mono` cannot be ported naively.


## Results (2026-09-20)

### Steps 1-3: the decomposer and the pluggable source (landed)

- `clean_spectra_utils.subformula_candidates(mz, formula, precursor_type, tolerance_da)` returns
  the ion **compositions** within a mass window, via a mass-pruned DFS over the same budgets,
  RDBE rule and electron correction the filter uses. It reproduces `_subformula_masses` exactly
  on every formula/adduct pair tested (~2,160 queries, 0 mismatches), and costs 0.076 ms/query
  on the real NIST20 distribution (p99 0.57 ms) where the full grid would be up to 1.35M masses.
- `isotope_utils` takes a `formula_source` callable instead of the `dag_formula_mzs` /
  `dag_formula_strs` / `dag_mz_tol` triple, with `dag_formula_source` and
  `subformula_formula_source` as the two adapters and `None` as the m/z-proxy fallback.
  `_fragment_expected_ratios` now returns one ratio per candidate and `_ratio_gate` applies the
  `median` / `all` / `any` rule. With a one-composition source all three rules agree, so the DAG
  path is unchanged by construction - and the 49 pre-existing isotope tests pass untouched.
- `protect_dag_mono` became `protect_known_mono`, since protection is now a statement about
  whatever source is configured.

### Step 1: the subformula grid is discriminating for ordinary molecules

`scripts/debug/fragnnet/debug_subformula_grid_density.py`, 2,915 NIST20 molecules, `[M+H]+`, FT 4 mDa.
Coverage is the share of random m/z that *some* subformula ion explains - the lower, the more
an explanation is worth:

| precursor mass (Da) | molecules | coverage mean | coverage p90 | candidates/query | ms/query |
|---|---:|---:|---:|---:|---:|
| ≤200 | 527 | 0.027 | 0.050 | 0.03 | 0.021 |
| 200-300 | 1,006 | 0.060 | 0.105 | 0.08 | 0.034 |
| 300-400 | 653 | 0.098 | 0.155 | 0.16 | 0.052 |
| 400-500 | 270 | 0.136 | 0.215 | 0.25 | 0.075 |
| 500-700 | 201 | 0.182 | 0.280 | 0.33 | 0.091 |
| 700-1000 | 199 | 0.351 | 0.445 | 0.89 | 0.237 |
| >1000 | 59 | 0.536 | 0.711 | 4.32 | 0.989 |

9.2% of molecules exceed coverage 0.25 (lightest 475 Da, median 810 Da) and 1.2% exceed 0.50
(lightest 971 Da). The Step 1 stop rule was 10% / 50%, so this passes - narrowly on the first,
comfortably on the second. Above ~700 Da a subformula explanation stops being evidence.

### Steps 0 and 4: measured against NIST `ann_isotopes`

`scripts/debug/isotopes/debug_dag_iso_peak_detection.py --frag-dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags
--n-samples 1500`. 1,500 FT spectra that carry at least one annotated isotope peak, 7,698
ground-truth isotope peaks, 68,128 peak decisions. Every arm sees the same spectra.

| Arm | TP | FP | FN | Prec | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Current (m/z proxy) | 6,151 | 1,468 | 1,547 | 0.807 | 0.799 | 0.803 |
| Group-f | 7,191 | 1,670 | 507 | 0.812 | 0.934 | 0.869 |
| DAG-native | 6,044 | 1,275 | 1,654 | 0.826 | 0.785 | 0.805 |
| **Subform-native** | 6,106 | 1,304 | 1,592 | **0.824** | **0.793** | **0.808** |
| DAG+Group-f | 7,201 | 1,513 | 497 | 0.826 | 0.935 | 0.878 |
| **Subform+Group-f** | 7,206 | 1,543 | 492 | **0.824** | **0.936** | **0.876** |
| DAG-native-protect | 1,059 | 198 | 6,639 | 0.842 | 0.138 | 0.237 |
| **Subform-protect** | 4,044 | 231 | 3,654 | **0.946** | 0.525 | 0.676 |
| Subform-infeasible (Step 0) | 4,949 | 3,375 | 2,749 | 0.595 | 0.643 | 0.618 |

**Gate (F1 within 0.01 of the matching DAG arm, precision no worse by 0.02): passed.** Native
+0.003 F1 / -0.002 precision; with group-f -0.002 / -0.002. The two sources agree on
**67,575 / 68,128 = 99.19%** of individual peak decisions, which is the number to quote: this is
not two methods that happen to score alike, it is the same method reading the same composition
from a different place.

**Step 0 did not close the plan, but it changed what the filter means.** `filter_subformula_peaks`
on its own already removes 64.3% of the annotated isotope peaks in this subset - below the 80%
stop rule, so a detector still buys something, but any run with `filter_subformula_peaks: True`
is already doing most of this work. It also removes 3,375 peaks that are not annotated isotopes,
so it is not a free win either.

**The plan's guess about protection was wrong, in the useful direction.** It predicted subformula
protection would be near-vacuous and DAG protection the meaningful one. Measured, DAG protection
destroys recall (0.138) because the DAG explains nearly every real peak, while subformula
protection is the **highest-precision arm measured here at 0.946**, at half the recall. If a
precision-first operating point is ever wanted, that is the arm - it was not reachable before.

Caveats on these numbers: the sample is restricted to FT spectra that carry NIST isotope
annotations (120,393 of 1,029,998 FT spectra), so the absolute rates run well above the
corpus-wide 0.700 / 0.733 operating point in
[detect_isotope_peaks_tuning.md](../reference/detect_isotope_peaks_tuning.md). The arm-to-arm
comparison is like for like, which is what the gate needed. Subformula arms use a 4 mDa window
against the DAG arms' 15 mDa, because the two tolerances mean different things: one matches a
peak to a node, the other decomposes a mass.

### Step 5: config surface (landed, still off)

Every `config/*_template.yml` now declares:

```yaml
  remove_isotope_formula_source: none  # ["none", "subformula", "dag"]
  remove_isotope_candidate_rule: median  # ["median", "all", "any"] when subformulas disagree
  remove_isotope_subformula_mz_tol:  # blank = instrument tolerance
  remove_isotope_protect_known_mono: False
```

`remove_isotope_use_dag_formula` and `remove_isotope_protect_dag_mono` stay declared so older
configs and checkpoints resolve, but setting either to True now **raises** at dataset setup
naming its replacement, rather than being silently ignored. `frag_dp` was not re-added to any
non-FraGNNet template; `formula_source: dag` is the only path that needs it.

### What is left

- Step 6 (soft output) is unstarted and unchanged: the candidate set is a posterior, the mask is
  a hard label, and the recorded position is that isotope peaks cannot be confidently detected.
- Nobody has trained with `remove_isotope_peaks: True`. The gate says the subformula source is as
  good as the DAG source; it says nothing about whether removing isotope peaks from targets helps
  a model. That is still the open question, and it predates this plan.

## Steps

Steps 0-5 are done; this section is the design as executed, and Results holds what came back.
Step 6 is untouched.

### Step 0 — does the existing subformula filter already remove isotope peaks? (done)

`spec_params.filter_subformula_peaks` already drops peaks with no feasible subformula at
`subformula_ppm` / instrument tolerance. A ¹³C peak sits +1.003355 Da from its M+0, and the
nearest chemically reachable alternative (+H, 1.007825) is 4.47 mDa away — outside FT tolerance
below roughly m/z 1000. So the filter may already be removing most labeled isotope peaks, for
free, in every run that enables it.

- Measure: on FT NIST20 with `ann_df.ann_isotopes` as ground truth, the fraction of labeled
  isotope peaks already removed by `filter_subformula_peaks`, by m/z decile and by precursor size.
- **Stop rule:** if ≥ 80% of labeled isotope peaks are already removed by the filter, close this
  plan `CLOSED NULL` and instead document the subformula filter as the de facto isotope-removal
  mechanism. A second detector buying the last 20% is not worth a new code path.

### Step 1 — candidate-grid feasibility over the real formula distribution (done)

Extend the preliminary table above to all FT NIST20 training molecules: candidate count,
enumeration time, and 4 mDa coverage fraction per precursor.

- Verify: coverage-vs-precursor-mass curve, plus the mass at which coverage crosses 0.25.
- **Stop rule:** if more than 10% of FT molecules sit past that crossing, the full-grid form is
  out and Step 2 must be the bounded decomposer (it is written that way regardless); if more than
  half do, the whole approach is not discriminating enough — stop and record the curve.

### Step 2 — a composition-returning decomposer (done)

Add `subformula_candidates(mz, formula, precursor_type, tol) -> list[dict[str, int]]` to
`clean_spectra_utils.py`, beside the budget logic it reuses.

- Implementation: depth-first walk over elements with a prefix-mass bound, pruning any branch
  whose remaining-mass interval cannot reach the window — **not** the full cartesian product that
  `_budget_ion_masses` builds. Cost then scales with the number of candidates in the window, not
  with the size of the grid, which is what keeps the peptide-sized case affordable.
- Reuse `_precursor_ion_budgets`, `RDBE_COEFFICIENTS`, `_precursor_ion_charge`, and the electron
  correction unchanged, so the filter and the detector cannot disagree about what a feasible ion is.
- Do not refactor `_subformula_masses` or `filter_subformula_peaks`; they stay as they are.
- Tests (`tests/test_clean_spectra_utils.py` or a new `tests/test_subformula_candidates.py`):
  masses of the returned compositions agree with `_subformula_masses` on small formulas; CMF
  budgets (`[M+Na]+` → `[frag+H]+`, `[M+Cl]-` → `[frag-H]-`); negative-RDBE rejection; empty
  window; tolerance boundary at exactly `tol`; an unsupported element and an unsupported adduct
  raise.

### Step 3 — make the formula source pluggable in `isotope_utils` (done)

- Replace the `dag_formula_mzs` / `dag_formula_strs` / `dag_mz_tol` argument triple in
  `_fragment_expected_ratio`, `detect_isotope_peaks_for_training` and
  `detect_isotope_peaks_for_training_cleanup` with one `formula_source` callable
  `(m0_mz) -> list[dict[str, int]]`, plus adapters: `dag_formula_source(...)` wrapping today's
  `_find_dag_formula`, `subformula_formula_source(...)` wrapping Step 2, and `None` keeping the
  m/z-proxy fallback. One code path, three sources, no behavior change for the DAG source.
- `_fragment_expected_ratio` returns `(ratio, spread)`. Decision rules to implement behind one
  argument and settle in Step 4:
  - **A (point)**: median expected ratio over candidates — closest to today's DAG behavior, and
    identical to it whenever the candidates agree, which the preliminary measurement says is the
    median case.
  - **B (intersection)**: flag only if the observed ratio is consistent with *every* candidate.
    Highest precision, lowest recall.
  - **C (union)**: flag if consistent with *any* candidate. This is what a naive port does, and it
    can only loosen the gate — expect it to lose precision.
- `protect_dag_mono` does not port. Its subformula analogue is "the peak is subformula-feasible",
  which Step 1 shows is near-vacuous for large precursors. Implement instead as: protect only when
  the peak is subformula-feasible **and** its m/z is not reachable as M+k of any observed lower
  peak within tolerance. Keep it off by default and measure it as its own arm.

### Step 4 — evaluate against `ann_isotopes` (done, gate passed)

Add `subformula_native`, `subformula_protect` and `subformula_formula` arms to
[`scripts/debug/isotopes/debug_dag_iso_peak_detection.py`](../../scripts/debug/isotopes/debug_dag_iso_peak_detection.py),
which already runs the DAG arms against `ann_df` on FT NIST20. Same sample, same protocol as the
recorded operating point (prec 0.700 / rec 0.733 / F1 0.716 at `min_coiso_fraction=0.05`; see
[detect_isotope_peaks_tuning.md](../reference/detect_isotope_peaks_tuning.md)).

Report per arm: TP/FP/FN, precision, recall, F1; and head-to-head against the matching DAG arm:
formula-agreement rate, decision-flip rate, and where the flips concentrate in m/z and precursor
size.

- **Gate:** adopt the subformula source only if F1 is within 0.01 of the matching DAG arm **and**
  precision is not worse by more than 0.02. Otherwise mark the plan `STOPPED` with the table.
- Note what this evaluation cannot settle: `ann_isotopes` is NIST's own annotation, and the
  recorded position is that isotope peaks cannot be confidently detected at all. A subformula arm
  that ties the DAG arm ties it at prec 0.700, which is not an operating point anyone has agreed
  to ship.

### Step 5 — config surface (done)

Only after the Step 4 gate passes:

- Add `remove_isotope_formula_source: none | subformula | dag` and
  `remove_isotope_subformula_ppm`, defaulting to `none` / the instrument tolerance.
- Keep `remove_isotope_use_dag_formula`, `remove_isotope_dag_dp`, `remove_isotope_dag_compressed`
  and `remove_isotope_protect_dag_mono` declared and ignored, marked deprecated in the YAML
  comment. Removing them breaks `find_unknown_config_keys` for historical configs and checkpoints.
- **Do not re-add `frag_dp` to any non-FraGNNet template.** Removing that dependency is the point
  of the plan.
- The `NOT HOOKED UP` banner comes off only when the user decides to re-enable the feature, which
  is a separate decision from this plan.

### Step 6 — soft output (optional, not started)

The candidate set is a posterior over compositions, so the natural output is a per-peak score, not
a mask. The recorded position — hard labels from m/z-only many-to-one matching are the wrong
target, and isotope peaks cannot be confidently detected — argues for emitting a score and letting
the consumer downweight, with the hard mask off by default. Design this only if Step 5 lands.

## What this does not fix

The three measured blockers in
[detect_isotope_peaks_tuning.md](../reference/detect_isotope_peaks_tuning.md) are orthogonal to
the formula source; two of them survive this plan untouched:

1. `isotope_offsets_from_formula` uses `CoarseIsotopePatternGenerator`, so M+2 is reported at
   2.00671 for every formula and the real Cl/Br M+2 (1.99705 / 1.99795) falls outside FT's 4 mDa
   window. Unchanged here.
2. The cleanup path calls `detect_isotope_peaks_formula_aware` with
   `intensity_ratio_lo=0.0, intensity_ratio_hi=inf`, i.e. the intensity discriminator switched
   off. Unchanged here.
3. `_expected_mk_ratio` mass-scales molecular atoms by `mol_scale`, which is wrong for Cl/Br.
   This plan **does** fix it wherever the candidate set resolves the halogen count, the same way
   a DAG formula does today.

Also out of scope: the prediction-side `frag_params.num_isotopes` axis, QTOF (the detector is
FT-gated by design), and any change to `filter_subformula_peaks` itself.
