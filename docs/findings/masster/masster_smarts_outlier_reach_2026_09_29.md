# MASSter SMARTS rules and outlier spectra

> **Status:** FINDING · **Updated:** 2026-09-29
> The registered rules matter only on a small set of spectra, where much of the target
> intensity sits at m/z that no ordinary fragment reaches. On `i1`'s `[M+H]+` val set those are
> 37 spectra of 14 molecules, with mean cosine 0.25-0.56. They are moderately bad spectra, not
> the floor: the rules cannot raise `i1`'s lower tail (p5 moves at most 0.0001). The whole-val
> cosine ceiling moves by 0.0001. `[M-H]-` is untested. Model-free; no model trained.

## Question and answer

The user's goal for the SMARTS rules is to help outlier spectra, not the average. The question is
therefore not whether the rules help NIST20 on average, but whether they reach spectra the
model gets badly wrong, and how many.

- **Averaged over the dataset, the rules are negligible.** Summed over all NIST20
  `[M+H]+`/`[M-H]-` spectra, the 14 registered rules' rule-only products explain 0.033% of
  `[M+H]+` and 0.160% of `[M-H]-` target intensity beyond random (the 2026-09-27 gate rows).
- **On outliers, they reach real, unpredictable intensity.** Across NIST20, counting all splits,
  a rule-only product explains:

  | Share of spectrum intensity | `[M+H]+` spectra (molecules) | `[M-H]-` spectra (molecules) | Random null, `[M+H]+` / `[M-H]-` |
  |---|---:|---:|---:|
  | ≥ 10% | 361 (89) | 606 (148) | 18 / 8 |
  | ≥ 25% | 123 (36) | 327 (106) | 4 / 1 |
  | ≥ 50% | 23 (13) | 162 (78) | 2 / 0 |

- **These are `i1`'s outliers, and `i1` cannot fix them.** On `i1`'s `[M+H]+` val set:
  - 37 spectra (14 molecules) have ≥10% of their intensity at rule-only m/z;
  - `i1` puts no predicted intensity there, since it has no candidate at those formulas;
  - their cosine falls as the share rises (table below), and 11 of the 37 sit in the bottom 10%
    of val cosine (below 0.334).
- **Rules involved:** on those 37 spectra, primary sulfonamide O transfer (20), aryl sulfonate
  SO2 (9) and secondary aryl sulfonamide SO2 (8). On NIST20 `[M-H]-`, the same sulfonyl
  family dominates the tail (sulfonamide SO2, sulfonate SO2, primary sulfonamide SO2).

`i1` val `[M+H]+`, template-bearing spectra by rule-only share of linear intensity:

| Rule-only share | Spectra | Molecules | `i1` mean cosine | Cosine cap without the rule peaks |
|---|---:|---:|---:|---:|
| 0 | 1,722 | 146 | 0.725 | 1.000 |
| 0-1% | 102 | 37 | 0.665 | 1.000 |
| 1-5% | 133 | 32 | 0.687 | 0.999 |
| 5-10% | 39 | 17 | 0.750 | 0.989 |
| 10-25% | 31 | 10 | 0.555 | 0.950 |
| 25-50% | 5 | 3 | 0.396 | 0.763 |
| > 50% | 1 | 1 | 0.247 | 0.441 |

The whole val set has 52,020 spectra with mean cosine 0.762. The cosine cap is
`sqrt(1 − s²)`, where `s` is the rule-only share of the spectrum's L2 norm on linear intensity.
It is the best cosine a model can reach while it has no candidate at those m/z. Lifting every
cap to 1 raises the whole-val mean ceiling by 0.00008.

## The lower tail: the rules do not reach it on `[M+H]+`

The user's case for the rules is that they raise the lower bound, the worst spectra. This was
tested on `i1`'s val set with an optimistic upper bound: each spectrum's cosine gains
`1 − cap`, the part it cannot reach without rule candidates.

- **The worst spectra carry no rule-only intensity.** None of the bottom 500 val spectra
  (cosine ≤ ~0.02) has ≥10% rule-only intensity, and none of the worst 100 molecules does
  either.
- **The low quantiles barely move, even under the optimistic bound:**

  | Val cosine quantile | `i1` now | With rules, upper bound |
  |---|---:|---:|
  | p1 | 0.0223 | 0.0223 |
  | p5 | 0.1786 | 0.1787 |
  | p10 | 0.3344 | 0.3346 |
  | Molecule-mean p1 | 0.2085 | 0.2085 |

- **Rule-only peaks are a minority of their own spectra's error.** They account for 18.8% of
  the cosine gap (`1 − cos`) on the 37 spectra with ≥10% rule-only intensity. `i1` gets the
  rest of those spectra wrong for other reasons.
- **So on `[M+H]+` the rules target moderately bad spectra (cosine 0.25-0.56), not the floor.**
  Whatever drives the near-zero-cosine spectra, these rules do not touch it.
- **`[M-H]-` is untested.** Its rule-only tail is larger (162 spectra at ≥50% rule-only
  intensity, against 23 in `[M+H]+`), and no `[M-H]-` model exists to rank its spectra.
  Whether the rules reach the `[M-H]-` floor needs that model's val cosines.

Log: `output/smarts_outliers_2026_09_29/lower_tail_log.txt`.

## What this means for judging SMARTS arms

- **A whole-split metric cannot show the effect.** Even a perfect use of the rules moves the
  `[M+H]+` val ceiling by about 0.0001, far below the ~0.004 seed spread.
- **The effect must be read on the template-bearing subset and its tail.** For example, cosine
  on spectra with rule-only share ≥10%, compared with a templates-off control on the same
  spectra.
- **The tail is about four times as frequent in `[M-H]-`:** 0.39% of its spectra reach ≥10%
  (606 of 156,532), against 0.10% in `[M+H]+` (361 of 367,379). `[M-H]-` has no trained model
  yet.
- **G0 in the SMARTS pathway plan (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_smarts_pathway_plan.md`)** gates
  `t1i` on the whole-split coverage ceiling. That gate measures the average, not the outliers
  this goal targets.

## Method and limits

- **Rule-only m/z:** the per-class gate's unique channels. These are the channels of a rule's
  disconnected product that no connected atom set of the same heavy-atom composition reaches
  within 4 H, matched at 10 ppm with a 200 m/z floor. Rules are the 14 registered ones, each in
  its admitted polarity; per spectrum, the union over rules.
- **Dataset-level and tail counts** come from the 2026-09-27 gate rows
  (`output/smarts_reach_2026_09_27/rule_gate*.pkl`). There each rule's rows follow the
  molecule's spectrum order, so a spectrum's value is the maximum over its rules, a lower bound
  on the union.
- **`i1` join:** from `output/ev_ladder_census_2026_09_28/i1_val.pkl` (per-spectrum `cos_ref`
  and peaks). The cap uses linear intensity, not the metric's own transform, so it is
  approximate.
- **Coverage, not model gain:** a templates-on model still has to rank those candidates. The
  arm is the test.
- **Artifacts:** `output/smarts_outliers_2026_09_29/`, holding `i1_rule_outliers.py`, its log
  and `i1_val_rule_outliers.pkl`.
