# SMARTS pathway gates G0, G1, G2, G2b: every arm fails its gate

> **Status:** FINDING · **Updated:** 2026-09-30
> Phase 1 of the SMARTS pathway plan (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_smarts_pathway_plan.md`), run
> 2026-09-29/30 on the local workstation. The decision rules were fixed before any result, and
> two adversarial verifiers per gate recomputed the headline numbers from the artifacts. They
> found no blocking issue.
>
> - **Scope:** NIST20 only. MONA is excluded (user decision, 2026-09-30).
> - **Registry:** 14 active rules. The vocabulary stays at 22.
> - **Test split:** never inspected.

## Decision

| Arm | Gate | Pre-registered rule | Result | Verdict |
|---|---|---|---|---|
| `t1i` on `[M-H]-` | G0 | whole-split coverage-ceiling rise ≥ +0.003 | +0.00125 [0.00091, 0.00163] | **FAIL** |
| `p1` (Track A) | G1 | headroom ≥ +0.003 and `rules` over `context` CI lower > 0 | headroom −0.000012 [−0.000040, +0.000013]; gain +0.000026 [−0.000020, +0.000084] | **STOP** (CI half-widths ≪ 0.003, so not underpowered) |
| `p1c` (Track B) | G2/G2b | promoted families + model operators reach a ceiling ≥ +0.003 | 0 families promoted, 0 operators | **FAIL** |

`t1i` on `[M+H]+` ran only as `p1`'s control, gated on "G1 continues", so it is not warranted
either. No Phase 2 arm of the plan has a gate that passes.

## Scope and units

- **Coverage ceiling.** Per spectrum, w is the eval-convention target: L1-normalised, linear,
  unbinned. With N = Σw², A0 is the w² the baseline covers and dA the w² newly covered. The rise
  is √((A0+dA)/N) − √(A0/N), averaged over every spectrum of the stratum (spectra with nothing
  new contribute 0). CIs are a molecule-clustered bootstrap.
- **Candidate-space, not W&B.** The rise is an upper bound on the `cos_sim_0.01` gain, not the
  W&B metric.
- **Production channels.** Template sides use cut count `len(broken_bond_indices)` and
  `formed_bond_count`, the valence bound, ±4 H and no CMF. This is the channel function the
  Phase 0 code uses in the model.
- **Unique coverage.** A peak counts only if no connected atom set of the same heavy-atom
  composition reaches it within ±4 H (`reachable_h`, cap 200,000). Capped searches earn no
  credit.
- **`[M-H]-` molecule assignment.** From `data/split/nist20_ma_mi_inchikey`, via 2D InChIKey,
  restricted to model-free use. That split and `i1`'s `nist20_inchikey` are independent draws.
  Of the `ma_mi`-val `[M-H]-` molecules that also have `[M+H]+` spectra, 1,108 are in `i1`'s
  train. So nothing trained on `i1`'s split may be scored on it (open decision D1).

## G0: `[M-H]-` templates on

**Headline stratum:** NIST20 OB/OB_F `[M-H]-` spectra, train, val and test pooled.
- 121,053 spectra from 10,833 molecules.
- 6,823 spectra (5.6%) from 568 molecules carry an eligible template.
- Only 1,360 spectra (1.1%) gain anything.

| Stratum | Spectra | Rise [95% CI] | Rise on template spectra (D2) |
|---|---:|---|---:|
| **Headline** | 121,053 | **0.00125 [0.00091, 0.00163]** | 0.0222 |
| train | 72,310 | 0.00125 [0.00079, 0.00178] | 0.0208 |
| val | 24,540 | 0.00141 [0.00073, 0.00216] | 0.0303 |
| test (census only) | 24,203 | 0.00109 [0.00049, 0.00188] | 0.0197 |
| NIST20 QTOF | 2,394 | 0 | 0 |

**Bounds.**
- The baseline is a stand-in pool: all 0-, 1- and 2-cut components. It over-covers, with A0/N
  a median 0.99 on template spectra.
- Without it, the rise is at least dA/(2N) = 0.00080.
- A fully pool- and twin-free upper bound, √(dA/N) with A0 = 0 and no twin exclusion, is
  **0.00269** (verifier, point estimate). The gate fails whatever the real pool is.
- The random disconnected null gives 3×10⁻⁷.

**Sensitivities.** The headline stays within 0.00011:
- no twin exclusion: 0.00136;
- optimistic valence: 0.00125, since the formed-bond correction never binds here;
- capped sides credited: 0.00125.

**Concentration.** The rise is heavy-tailed: 100 spectra hold 58% of it.

**Per rule** (rule alone, headline stratum):

| Rule | Molecules | Rise | L1 pp on its spectra |
|---|---:|---:|---:|
| `aryl_sulfonamide_so2` | 192 | 0.00059 | 3.95 |
| `aryl_sulfonate_so2` | 85 | 0.00031 | 6.08 |
| `primary_aryl_sulfonamide_so2` | 51 | 0.00028 | 6.21 |
| `n_acyl_aryl_sulfonamide_carbonyl_o_transfer` | 7 | 0.00005 | 7.26 |
| `n_nitramine_n2o` | 2 | 0.00002 | 16.29 |
| `tertiary_aryl_sulfonamide_so2`, `n_phenylbenzamide_co_smiles`, `sulfoxide_so_loss` | 53 / 173 / 16 | ≈ 0 | ≈ 0 |
| `diacylhydrazine_carboxylate_o_transfer`, `pfas_sulfonate_f_transfer` | 0 | — | — |

- The two largest `[M-H]-` per-class effects in the plan's §1 (diacylhydrazine +49.6, PFAS)
  come only from MONA spectra. With MONA excluded, they have no evidence.
- Validation: with optimistic valence and no pool, the scorer reproduces the 2026-09-27
  per-class gate's `unique_fraction` exactly on 1,005 spectra.

## G1: rule provenance on `i1` (`[M+H]+`)

- **Run:** `i1` epoch 59 on val (52,020 spectra), with a fresh extract that saves candidate m/z
  and rule columns.
- **Primary marker:** `rule_any`, a candidate peak at a valid production channel of either side
  of an eligible `[M+H]+` template (1e-5 Da).
- **Coverage:** 187 val molecules (4.4%) and 2,598 spectra carry a template. After the save
  floor, 1,230 marked rows in 611 spectra and 83 molecules remain.
- **Target mass:** marked peaks hold 0.000225 of target mass per spectrum, and `i1` already
  predicts 0.000240 on them.

| Family (cross-fitted, candidate-space cos) | MLP over `context` [CI] | Linear over `context` [CI] |
|---|---|---|
| `existing` | +0.000589 [+0.000120, +0.000962] | +0.000191 [−0.000139, +0.000435] |
| `rules` | +0.000026 [−0.000020, +0.000084] | +0.000001 [−0.0000005, +0.0000038] |
| `rules+existing` | +0.000615 [+0.000184, +0.000949] | +0.000192 |
| `rules_permuted` (control) | −0.000006 | −0.000018 [−0.000034, −0.000005] |

- **Ceiling.** The literal swap of every marked peak to its target share gains only +0.000019
  whole-split before control subtraction, about 150× below +0.003.
  - The matched control gains more (+0.000031), so the headroom is slightly negative.
  - The sign partly reflects matching on m/z band and target rank but not on predicted share.
    The decision does not depend on it.
- **Informative marker.** `rule_disc`, twins of a disconnected template side (30 molecules),
  has headroom +3×10⁻⁹ [−2.3×10⁻⁵, +2.1×10⁻⁵].
- **Scope.** G1 tests flags on rule peaks that `i1` already produces as candidates. Rule
  products `i1` cannot produce are G0's and G2's question.

## G2 and G2b: `[M-H]-` operators on fragments

**Setup.** Model-free: the stand-in pool replaces a checkpoint pool. Strict (production-matched)
valence is primary. The baseline is the pool plus the eligible registered template products
plus the generic whole-precursor bridge products.

**G2, rule-restricted continuations.** X is limited to the bridge-shaped registered motifs plus
aryl nitro NO. Train used all 456 motif-bearing train molecules (1,810 spectra); val used 136
molecules (537 spectra).
- **Selection:** the frozen rule selected **0 families**.
  - Four families beat the null on train: nitro ×2 (+1.20, +1.55 pp), sulfonamide NH (+3.86)
    and sulfonate OH (+3.44), strict lower limits > 0.
  - All four were rejected by the no-capped-searches rule: 60–81 capped real sets each.
  - At a cap of 1,000,000 the selection is still 0.
- **Post-hoc, not decision-bearing:** a cap-tolerant Bonferroni check confirms all four on val,
  but their load is 26–41 formulas per bearing spectrum, against a limit of 10.
- **Ceiling (descriptive, every motif family):** val 0.00104 [0.00051, 0.00168], with a
  pool-free upper bound of 0.00332. Train is 0.00162.

**G2b, generic operators.** 512 random train and 512 random val molecules; val has 1,980 spectra.

| Operator | Strict diff vs null, val (pp) [CI] | Load (formulas / spectrum) | Any-cap spectra | Model operator |
|---|---|---:|---:|---|
| bridge (≤3 heavy atoms) | −0.033 [−0.205, +0.158] | 177 | 8.4% | no |
| graft (terminal O/F, ≤4 bonds) | +0.004 [−0.096, +0.108] | 50 | 4.5% | no |

- **Real hits equal random ones.** Real bridge products reach 0.411 pp of unique intensity,
  random disconnected sets 0.443 pp. So the operators' descriptive ceilings are mostly chance.
  - bridge 0.0018, graft 0.0010, union 0.0023 on val;
  - bridge 0.0062, union 0.0049 on train;
  - none is null-adjusted.
- **Graft null.** It is size-matched only and gets about 9% more channels than real grafts, a
  mild leniency. The CI is still ±0.10 pp.
- **Discovery candidates only.** Graft of the O next to an aryl nitro N has load about 6 and
  22 train / 18 val molecules, and it misses on caps. It needs a chemistry read before any rule
  is written.

**Protocol limits found.** Both are pre-registered, and only the user should change them.
- **Load rule:** as measured on the stand-in pool, which inflates it, ≤10 formulas per bearing
  spectrum fails every family and operator by 2.5–20×.
- **No-caps rule:** zero capped searches is effectively unsatisfiable for families with 40–55
  atom molecules.
- **Neither changes Track B.** Even every motif family together stays at a val ceiling of
  0.00104 under the stand-in baseline.

## What this does not show

- **`t1i` gain is untested, as intended.** G0 bounds what `t1i` could add to `[M-H]-` coverage.
  It does not measure a trained model.
- **The D2 read is still open.** The bearing-subset value is 0.022 for G0; whether a gain
  confined to template spectra is worth pursuing is the user's call.
- **G1 is `[M+H]+` only.** No `[M-H]-` checkpoint exists, and the five `[M-H]-`-only rules have
  no `[M+H]+` val molecules.
- **The stand-in pool bounds Track B only loosely.** It is not the real candidate pool, which
  would cover less.

## Artifacts and code

**Artifacts:** [`output/smarts_pathway_gates_2026_09_29/`](../../../output/smarts_pathway_gates_2026_09_29/).
- Reports: `g0_report.md`, `g1_report.md`, `g2_report.md`, `g2b_report.md`.
- Data: per-spectrum rows, fits, ceilings, families json, provenance.
- Slices: `mh_spec.pkl`, `split_map.csv`, `spec_split.csv`.

**Code** (all under `scripts/debug/`, CPU except the G1 extract):
- [`debug_masster_smarts_g0_ceiling.py`](../../../scripts/debug/smarts/debug_masster_smarts_g0_ceiling.py) (new).
- [`debug_masster_postcut_information_gate.py`](../../../scripts/debug/masster_memorization/debug_masster_postcut_information_gate.py):
  rule columns, `--template_bearing_only`, the rule families and a `ceiling` step.
- [`debug_masster_excision_continuations.py`](../../../scripts/debug/smarts/debug_masster_excision_continuations.py):
  polarity plumbing, model-free stand-in pool, motif restriction, graft, confirmation step and
  the ceiling unit.

**Tests:** `tests/test_debug_masster_smarts_g0_ceiling.py`,
`tests/test_debug_masster_postcut_information_gate.py`,
`tests/test_masster_excision_continuations.py` and
`tests/test_debug_masster_pathway_continuations.py` pass (104 tests in total).

**Provenance.** The runs used the uncommitted Phase 0 tree, and `HEAD` moved from `a7aff613` to
`9d55d4a0` during G2b, through another session's commit. Rule names and the gate scripts'
source hashes are identical across all runs.
