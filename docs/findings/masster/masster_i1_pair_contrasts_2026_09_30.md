# Conditional success and failure contrasts for i1's candidate choices

> **Status:** FINDING · **Updated:** 2026-09-30
> Step 2 complete: matched success/failure comparisons on 50,607 eligible spectra / 4,221
> molecules. Broad fragmentation difficulty repeats; familiarity is the only contrast that
> passes the primary chemical-choice replication gate. Strict fragment-only support is too
> sparse to identify a new chemical rule. Reused validation; no model training or correction fit.

## What the comparison establishes

**A repeated error pattern, not a new cut rule.** On two different fragments, the model fails
more often when the measured choice needs additional cuts, cuts a ring, or needs a larger
absolute H shift. Those associations repeat in both molecule halves after matching the recorded
conditions. They do not isolate which local chemistry causes the failure: demanding the same
cut-class alternatives leaves very little common support.

**Molecular familiarity is the clearest repeat.** Within condition-matched fragment-only
choices, maximum similarity to training >=0.6 is associated with **20.8 percentage points less
error** than similarity <0.6. The narrower chemical-choice comparison also retains a familiarity
association, but is overwhelmingly precursor-versus-fragment and must not be generalized to all
fragment competition.

### Fragment-only, matching conditions and relationship

33,776 eligible spectra have no intact-molecule candidate in either group. Each row below
uses its own common-support subset, so these are not whole-val changes or additive effects.
The comparison is the error rate when the measured target has the property versus when the rival
has it; for familiarity the comparison is >=0.6 versus <0.6 similarity to training.

| Contrast | Matched error difference, percentage points [pointwise 95%] | Molecules |
| --- | ---: | ---: |
| Target needs more boundary cuts | +25.7 [22.8, 28.7] | 2,016 |
| Target cuts a ring, rival does not | +21.4 [16.7, 26.0] | 888 |
| Target needs a larger absolute H shift | +16.6 [12.7, 20.6] | 1,323 |
| Target breaks an aromatic ring, rival does not | +23.2 [17.8, 28.6] | 670 |
| Target retains a larger atom fraction | -7.8 [-9.3, -6.2] | 4,015 |
| Molecule is more familiar to training | -20.8 [-22.6, -18.9] | 4,112 |

These directions pass q<.05 in both molecule halves in this broader, post-hoc fragment-only
read. They are correlated descriptions of fragmentation difficulty, not separate causal effects.
The energy contrast (>40 versus <=40 NCE) is +1.1 [-1.1, 3.4] points on fragment-only choices;
this analysis does not establish energy alone as the missing discriminator after these controls.

### Requiring comparable cut-class alternatives

- Both groups have a unanimous boundary-cut signature in **11,064 of 50,607** eligible spectra.
  Ambiguity and lack of overlap further reduce the matched comparisons. This selection favors
  simpler chemistry; missing support is not a null effect.
- For familiarity, fix the *ordered* target/rival cut signatures and mass orientation as well as
  instrument, energy, mass, observed separation and retained-atom relationship. There are 240
  supported strata / **1,414 molecules**. Matched error rates are **13.6% versus 20.9%**, a
  -7.3 [-9.5, -5.0] point difference. The two molecule halves give -5.2 and -9.1 points and
  pass the primary replication gate (discovery q=.0344, replication q=7.50e-7).
- **95.6% of this comparison's overlap weight includes an intact-molecule candidate.** The
  strictly matched fragment-only familiarity read has only 95 molecules; its two halves have
  too little common support to replicate (4 and 17 molecules). The broader fragment-only
  familiarity association above is better supported, but does not hold cut classes identical.
- The apparent strict cut-count/size/precursor effects are largely the same precursor contrast.
  Their discovery-half q=.0685 misses the full 22-test threshold. They are not three new rules.
- For two fragments with the same competing cut-class signatures, the cut-count contrast has
  only **12 molecules**, and the ring-cut contrast has none. No specific cut class passes the
  primary discovery/replication gate with adequate support.
- Single-structure pairs retain the strict familiarity effect (-7.2 points). Excluding rerun-
  sensitive rows gives -7.1 points overall; the discovery half's q=.0545 narrowly misses the
  threshold, although its direction/magnitude and pointwise interval remain similar. Thus the
  exact gate crossing is less robust than the familiarity association itself.

### Meaning for a training signal

This gives a concrete *diagnostic target*: discriminate competing fragmentation outcomes on
unfamiliar molecules, particularly when the dominant outcome requires more extensive
fragmentation. It does **not** justify globally rewarding more cuts, ring cleavage or H shifts.
The earlier [attribute-reweighting null](masster_i1_base_peak_errors_2026_09_28.md) already tested
simple feature corrections and found almost no gain beyond score calibration. No new loss,
head modification, or validation improvement is demonstrated here.

## Question and design

When comparable candidate choices occur on different molecules, what distinctions accompany
correct versus incorrect base-peak predictions? This follows the
[ambiguity-preserving table](masster_i1_fragment_pairs_2026_09_30.md) and retains the earlier
[chemistry-reweighting null](masster_i1_base_peak_errors_2026_09_28.md).

A hit's measured base and model winner are the same group, so the step-1 table alone cannot
provide a meaningful competing choice for controls. The exporter now has an opt-in
`--include_rival`: choose the highest-scoring rounded peak outside the measured base-peak group,
using the same definition for both hits and misses. Every channel of that rival is saved. This
is fresh annotated inference on `i1`@59, all 52,020 validation spectra, batch size 128.

- Exclude uncovered bases, unavailable rivals, tied measured bases and non-strict measured
  preferences. Retain failures and hits; no outcome-based balancing or hand-picked examples.
- Never pick a target structure or average its features using model probabilities. A directional
  contrast is defined only if all target alternatives exceed all rival alternatives, or vice
  versa. Equal constants have direction zero; overlapping uncertain ranges are excluded.
- Compare cut count, atom fraction, absolute H shift, ring cutting, aromatic break/loss, basic-site
  retention, precursor identity, and 12 existing cut classes. Also test familiarity to training
  (stored `novelty`, actually maximum fingerprint similarity) and NCE as contextual contrasts.
- Match instrument class, NCE bin (20/40/60/100 boundaries), similarity bin (.4/.6/.8), precursor
  mass bin (250/400/600), observed rival/base intensity-ratio bin (.25/.75), and undirected
  retained-atom relationship. Remove similarity/NCE matching when that variable is the exposure.
  Context contrasts also fix the sign of target minus rival m/z. In the strict chemical-choice
  read they fix the ordered target/rival cut signatures, not just the unordered pair: high
  familiarity must not be compared with low familiarity when the preferred chemistry reverses.
  This direction control was tightened after the initial descriptive read.
- The primary `same_choice` analysis additionally matches the unordered pair of exact boundary
  cut-class multisets, requiring consensus within each ambiguous group. It matches coarse
  chemical choices, **not identical molecular environments**. `conditions` is a broader read;
  single-structure pairs and stable-export rows are sensitivity analyses. A post-hoc
  fragment-only sensitivity excludes any pair with an intact-molecule candidate in either
  group, after the initial strict cut/size contrasts proved dominated by precursor choices.
- Collapse repeated spectra to a molecule's mean error within stratum and direction. Exclude a
  molecule from a stratum if it supplies both directions, so opposing preferences compare
  different molecules. Require at least two molecules on each side. Stratum weights are the
  smaller side's molecule count. Report the +direction minus -direction error-rate difference,
  with molecule-clustered sandwich intervals on fixed common-support strata/weights.
- Random molecule halves (seed 20260930) preserve all spectra of a molecule together. Discover
  effects with BH q<.05 and at least 50 molecules in half 0; require the same sign, q<.05 and
  at least 50 molecules in half 1. Apply BH over the 22 prespecified contrasts within each read.
  These halves test internal replication, not untouched evaluation after prior validation use.
- Fresh versus previous export discrepancies are flagged by outcome, group counts, winner m/z,
  and intensity-share agreement (`rtol=1e-4`, `atol=3e-6`). Do not splice historical scores into
  the new groups. Check the result after excluding flagged rows.

## Interpretation limits

The rival is selected by model scores, and the target is selected by observed intensity. This
study diagnoses conditional errors among those choices. It does not identify a causal chemical
mechanism, prove a missing encoder feature, or estimate the benefit of changing a ranking loss.
Matching improves comparability only on the recorded coarse attributes and shared support.
The previous attribute-reweighting null is not overturned by an error association.

## Artifacts, reproduction, and checks

- Export: [`output/i1_pair_contrasts_2026_09_30/val/`](../../../output/i1_pair_contrasts_2026_09_30/val/).
  Its `channels.jsonl` adds `rival_contributor`; all descriptors and scores come from the same
  forward. All 52,020 outcome labels retain the prior counts. Extraction took 300.2 seconds.
- Analysis: [`output/i1_pair_contrasts_2026_09_30/analysis/`](../../../output/i1_pair_contrasts_2026_09_30/analysis/).
  `features.pkl`/`.csv` contains eligibility, consensus descriptors, context bins, molecule-half
  IDs and stability flags. `matched_contrasts.csv` contains every planned test and sensitivity,
  including unsupported tests. `replication.csv`, `summary.json` and `provenance.json` record
  the decision and source identity. `matched_examples.csv` contains illustrative successful/
  failing molecule pairs from common-support strata; their outcomes were selected for
  illustration and they are not additional evidence.
- 42 focused tests passed, including mass-rounding boundaries, ambiguous zero-probability
  alternatives, repeated-spectrum invariance, opposite-direction molecule exclusion and
  full-family multiple-testing adjustment. Ruff and whitespace checks passed.
- The analysis audits every target/rival channel count, share sum, disjoint membership and
  molecule identity. All spectra map one-to-one to the metadata. Of 50,607 eligible rows,
  50,225 agree with the previous export within the recorded stability checks.
- An independent groupwise implementation reproduced the strict familiarity estimate to
  1e-14 and its 1,414-molecule / 240-stratum support. All molecules stay in one half.
  [`audit_effect.py`](../../../output/i1_pair_contrasts_2026_09_30/audit_effect.py) and
  `analysis/independent_audit.json` preserve this check.
- Existing model/config edits were left untouched. No full model suite, training run,
  correction fit or untouched-test evaluation was performed.

```bash
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python scripts/debug/masster_memorization/debug_masster_fragment_pairs.py \
  --checkpoint saved_ckpts/i1_inst_ob_obf_i50ig7rf/ckpt/model-epoch=059.ckpt \
  --custom_fp config/masster/mh_ft_memorization_route_2026_09_24/i1_inst_ob_obf.yml \
  --spec_fp output/i1_worst_predictions_2026_09_30/subset_spec_df_val_all.pkl \
  --output_dp output/i1_pair_contrasts_2026_09_30/val --batch_size 128 --include_rival

/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python scripts/debug/masster_memorization/analyze_masster_pair_contrasts.py \
  --source output/i1_pair_contrasts_2026_09_30/val \
  --metadata output/i1_worst_predictions_2026_09_30/val_summary.pkl \
  --reference_pairs output/i1_fragment_pairs_2026_09_30/val/pairs.pkl \
  --output output/i1_pair_contrasts_2026_09_30/analysis
```

Use a fresh export directory; the extractor refuses to overwrite channels. The analysis can be
recomputed from its unchanged inputs. The next scientific step would require a concrete source
of discrimination beyond these coarse attributes and evidence that it improves rankings on
molecules excluded from fitting. This study does not select such a signal.
