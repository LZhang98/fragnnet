# Post-cut information gate on `i1`: explicit fragment and charge descriptors add nothing

> **Status:** FINDING · **Updated:** 2026-09-29
> Inference and offline fitting only, on the local RTX 2070 SUPER. `i1_inst_ob_obf`
> (`i50ig7rf`) at epoch 59, all 52,020 val spectra (4,222 molecules). This is step 2 of the
> user's three-step plan, after the
> [wrong base-peak table](masster_i1_base_peak_errors_2026_09_28.md) found no single repeated
> missing distinction.
>
> - **Question.** Does an explicit description of each candidate after cutting carry information
>   `i1` does not use? Two descriptions are tested: its retained connectivity and exposed cut
>   sites, and its relation to the charge site.
> - **Test.** A per-candidate correction `logit' = log p + g(x)`.
>   - It is fitted on half the val molecules and scored on the other half, by `mol_id` parity.
>   - The context-only correction (`log p`, NCE, instrument, precursor m/z) reproduces the known
>     `P^γ` gain, +0.0053 cosine. Each descriptor family is read by its gain over that baseline.
> - **Neither family adds more than about +0.001 cosine on held-out molecules, and neither
>   stands out.** Over the baseline, in three fitting settings:
>   - post-cut connectivity (80 columns): −0.0008 (MLP), −0.0004 (linear), +0.0010 (regularized
>     MLP);
>   - charge (20 columns): +0.0002, +0.0001, +0.0009;
>   - the earlier eight structure attributes: +0.0005, +0.0002, +0.0008.
>   - The regularized baseline itself fits 0.0004 worse. Against the best baseline, no family
>     adds more than +0.0006.
> - **For step 3:** hand-built post-cut and charge descriptors cannot predict `i1`'s errors on
>   new molecules. This lowers the prior for a branch that describes the candidate after cutting,
>   without ruling it out: a learned branch trained on training molecules sees different features
>   and a different objective.

## Question

Step 1 found that `i1`'s wrong base-peak choices come from a broad transfer deficit. They are not
a systematic bias a global correction removes: eight attribute weights add +0.0004. A
per-attribute linear reweighting cannot use the chemistry of each candidate jointly, though.
Step 2 asks whether a richer, explicit description of each candidate after cutting predicts the
errors on molecules the correction never saw. If it does, the information exists and transfers
across molecules, and a representation branch has something to learn. If it does not, the
description adds nothing the model does not already use.

## Method

[`debug_masster_postcut_information_gate.py`](../../../scripts/debug/debug_masster_postcut_information_gate.py)
has two subcommands.

- **`extract`** runs the annotated eval-mode forward of the
  [base-peak probe](masster_i1_base_peak_errors_2026_09_28.md#method). Channels are merged into
  candidate peaks and the eval-convention target is mapped at 10 ppm.
  - For every candidate peak with predicted share ≥ 1e-5 or target intensity (119 per spectrum),
    it saves descriptors computed per fragment from its membership row. They are averaged over
    the peak's structures with the model's shares.
  - 7 min for the val split.
- **Descriptor families:**

  | family | columns | contents |
  | --- | ---: | --- |
  | basics (in every family) | 6 | H shift, its magnitude, intact precursor, heavy atoms, fraction of precursor atoms, cut count |
  | existing | 4 | ring bond cut, aromatic ring lost whole or broken, basic site kept (the eight step-1 attributes, with the basics) |
  | post-cut connectivity | 74 | retained bonds and the fragment's cyclomatic number; SSSR and aromatic rings kept and broken; conjugation kept; ring, aromatic and multiple bonds cut; retained and lost cut-site atom types (12 each); cation-stabilizing retained and lost cut sites (acyl, benzylic, α-heteroatom, allylic, tertiary carbon, heteroatom); cut-bond classes (20); element counts, H count and RDBE of the neutral species |
  | charge | 14 | basic site kept, kept matches, its class (7); graph distance from the retained cut sites to the kept basic site (and a missing flag); Gasteiger charge sum, minimum and maximum of the fragment, and the complement's sum |

- **`fit`** fits `logit' = log p + g([log p, context, x])` by listwise softmax cross-entropy
  against the target over the saved candidates.
  - `g` is zero at start: a two-layer MLP (width 128, dropout 0.1), or a linear map.
  - It is fitted on val molecules of one `mol_id` parity, with early stopping on 10% of those
    molecules held out, and scored on the other parity; then the folds are swapped.
  - Every spectrum is scored by a correction that never saw its molecule.
  - Reported: held-out mean cosine (candidate space, tolerance-matched, as in step 1), top
    candidate agreement, and molecule-clustered 95% intervals.
- The context-only family is the baseline. Its gain is the calibration gain, which
  [step 1](masster_i1_base_peak_errors_2026_09_28.md#5-without-rank-selection-the-model-is-calibrated-in-aggregate)
  measured as +0.0052 for `P^γ`.

## Result

Held-out mean cosine gain over the raw 0.7878, and over the context-only baseline:

| family | MLP gain | MLP over baseline [95%] | linear over baseline [95%] |
| --- | ---: | ---: | ---: |
| context only | +0.0053 [+0.0050, +0.0057] | — | — (linear gain +0.0049) |
| existing | +0.0059 | +0.0005 [+0.0001, +0.0009] | +0.0002 [−0.0001, +0.0004] |
| post-cut connectivity | +0.0045 | −0.0008 [−0.0015, −0.0001] | −0.0004 [−0.0009, −0.0001] |
| charge | +0.0055 | +0.0002 [−0.0003, +0.0007] | +0.0001 [−0.0002, +0.0003] |
| post-cut + charge | +0.0035 | −0.0018 [−0.0026, −0.0010] | −0.0004 [−0.0009, −0.0001] |
| all | +0.0036 | −0.0017 [−0.0027, −0.0006] | −0.0005 [−0.0009, −0.0001] |

- **Top-candidate agreement does not move:** 0.614 raw, 0.614 for the baseline, and 0.610-0.615
  for every family across the three fits.
- **The inner early-stopping loss agrees.** It is the listwise cross-entropy on unseen molecules
  inside the fit half. The best inner loss is 2.3003 / 2.3382 for the baseline and
  2.2981 / 2.3474 for post-cut (the two folds). No family ends below the baseline.
- **Regularized MLP** (width 32, dropout 0.3, weight decay 1e-3). The baseline gain is
  +0.0049 [+0.0046, +0.0052]. Over it:
  - existing: +0.0008 [+0.0005, +0.0010];
  - post-cut: +0.0010 [+0.0005, +0.0015];
  - charge: +0.0009 [+0.0006, +0.0012];
  - post-cut + charge: +0.0006 [−0.0000, +0.0011];
  - all: +0.0006.

  The total gains, 0.0055-0.0059, sit within +0.0006 of the default MLP's baseline (+0.0053).

## Reading

- **The explicit description of the candidate after cutting carries almost no information `i1`
  fails to use on new molecules.**
  - The post-cut family covers retained connectivity, exposed cut sites and their chemistry,
    cation-stabilizing sites, bond classes and ion composition.
  - Learned on 2,100 val molecules, it predicts `i1`'s errors on the other 2,100 at most about
    0.001 better than `log p` and context do.
  - That small amount is not specific to post-cut connectivity: the eight basic attributes and
    the charge family reach the same.
  - A linear map and two MLP settings agree, so the null is not a fitting artifact.
- **Charge-site relations add no more,** which agrees with step 1: charge-side decisions are 1.5%
  of val top-two pairs.
- **What this does and does not say about step 3.**
  - The gate asks whether hand-built descriptors predict how `i1` errs on new molecules. A branch
    trained on training molecules faces a different problem: it must change what the model learns,
    where the model already fits the training molecules.
  - A gain here would have shown the information exists and transfers. The null says these
    descriptors do not carry it. A learned post-cut encoding could still find features they
    miss, but the prior for it is lower than before.
  - The screen is the direct test.

## Limits

- One checkpoint, one seed; the correction is fitted on 2,100 molecules per fold.
- Descriptors of peaks with several structures are averaged with the model's structure shares
  (median 2 structures per peak).
- The correction sees only the saved candidates (≥ 99.8% of predicted mass per spectrum).
- The objective is listwise cross-entropy; the reported metric is cosine.

## Artifacts

`output/postcut_gate_2026_09_29/` (Git-ignored):
- `run_gate.sh`;
- `i1_e059_val.npz`, 2.5 GB, with provenance;
- `fit*.csv`, `fit*.log` and per-spectrum tables.

The script's `--hidden 0` (linear) option and its Black formatting came after the extraction and
the default fit. The extraction path and the MLP are unchanged, so the provenance source hash of
the extraction differs from the file only by those edits.

Tests: `tests/test_debug_masster_postcut_information_gate.py` (8).
