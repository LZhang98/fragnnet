# NIST suspicious-spectrum human review

> **Status:** GUIDE &nbsp;·&nbsp; **Updated:** 2026-08-03
>
> How to run the suspicious-spectrum review queue.

Generate a ranked queue and visual report:

```bash
python scripts/eval/nist_bad_data_review.py \
  --spec_fp data/proc/nist20/spec_df.pkl \
  --mol_fp data/proc/nist20/mol_df.pkl \
  --uwpr_feasibility_csv \
    data/uwpr/nist20_uwpr_hits_feasibility.csv \
    data/uwpr/nist20_uwpr_hits_feasibility_d4.csv \
  --out_csv data/eval/nist20_bad_data_review.csv \
  --out_html data/eval/nist20_bad_data_review.html \
  --out_pdf data/eval/nist20_bad_data_review.pdf
```

The HTML and PDF default to the top 500 ranked candidates. Each PDF review page uses the same
two-column CE-spectrum grid as `plot_ce_trend_anomaly_groups.py`, with a shared header and molecular
structure. The header reports recorded precursor m/z, computed precursor m/z, and precursor error
in ppm when the precursor type is supported. When `substantial_cross_ce_orphan_intensity` is
present, the grid plots one representative spectrum per other collision-energy bin, capped at the
11 nearest bins. Those panels are sorted from low to high using the detector-selected NCE or ACE
value, with the review target inserted at its corresponding CE and highlighted with a pale
background and border.
Cross-CE panels fill each column from top to bottom before continuing in the next column.
Candidates sharing molecule ID, precursor type, and instrument type are packed onto the same page,
up to 12 panels, when none requires a dedicated comparison layout. They are ordered from lower to
higher collision energy, using NCE for FT spectra when available and ACE otherwise; missing CE is
last. Each candidate remains separately highlighted and labeled. Mirror, same-CE replicate, and
cross-CE comparison pages remain dedicated to one candidate so their evidence panels stay
unambiguous. Dedicated pages show at most 11 total comparison panels; formula conflicts are kept
first, followed by same-CE replicates and then other-CE spectra.
For `replicate_consensus_disagreement`, every valid same-CE peer used by the consensus calculation
is plotted on the candidate page.
For `substantial_uwpr_contaminant_only_intensity`, D3 `dag_feasible` peaks are green and labeled
`D3 DAG`, peaks that become feasible only in D4 are purple and labeled `D4+ DAG`, and peaks called
`contaminant_only` by every source returning a decision are orange and labeled `UWPR`; undecidable
sources abstain. Only the orange consensus category supports bad-data priority; DAG-feasible
matches are context.
For `duplicate_spectrum_formula_conflict`, the grid plots every exact duplicate partner and labels
its molecule ID, formula, adduct, recorded and computed precursor m/z, and molecular structure. The
comparison uses a mirror plot with the primary spectrum upward, the conflicting assignment
downward, and their cosine similarity in the title. The page labels the primary record `REVIEW
TARGET` and its comparison `CONFLICTING ASSIGNMENT`; these roles are relative to that page and do
not assert which record is incorrect. Target and conflicting molecular structures are placed in a
reserved header strip so they never cover spectrum peaks.
All plots on the page use the same m/z scale, start at the top-left, hide unused grid cells, and
label the available ACE and/or NCE collision energy. Non-cross-CE grids fill left-to-right;
cross-CE grids use the column-major order described above. A fixed-height header keeps the
spectrum grid near the top of every page. A dashed vertical line marks recorded precursor m/z,
falling back to computed precursor m/z when the recorded value is unavailable; mirror plots use
one combined marker within 0.01 Da and separate blue target and red conflict markers when the
values differ more substantially.
Every plotted spectrum title reports D3 and D4 `wRecall`, the fraction of its observed intensity
matched by the corresponding DAG under the instrument-specific m/z tolerance. The default D3
wRecall build is the same MBFS build used by `uwpr_feasibility.py`, so green D3 peak annotations
and the displayed D3 score use one DAG definition.
Candidates with D3 or D4 `wRecall >= 0.75` are marked `dag_wrecall_gated` when they have only
soft spectral evidence. They remain in the CSV for auditing but are omitted from HTML/PDF review.
Hard conflicts, record-integrity, metadata-completeness/consistency, and label-ambiguity evidence
are exempt because DAG explainability cannot refute those defects. Low or unavailable D3/D4
`wRecall` does not create a bad-data reason.
The PDF defaults to `[M+H]+`, `[M-H]-`, `[M+Na]+`, `[M+K]+`, `[M+Cl]-`, and `[M]+`; change the list
with `--pdf_prec_types` and the limit with `--max_pdf_spectra`.

Candidates with hard conflicts are shown first, followed by candidates supported by the most
independent evidence families and then by priority score. The score is not a probability and must
not be used directly as a training filter. The CSV preserves separate hard and heuristic reasons
and includes blank reviewer fields. Record one of `keep`, `exclude`, or `needs_followup` in
`review_status`; never edit the raw NIST tables during review.

## Signals

Higher-confidence consistency checks:

- malformed, non-finite, negative-intensity, or zero-total-intensity peak arrays;
- precursor-adduct polarity inconsistent with `ion_mode`;
- the same exact-export spectrum with at least three peaks assigned to molecules with different
  formulas;
- missing molecule metadata.

Review heuristics, not proof of bad data:

- precursor m/z inconsistent with molecular exact mass and the assigned adduct;
- missing precursor m/z;
- no collision-energy metadata or negative collision energy;
- at most two peaks;
- at least 5% of intensity more than 4.5 Da above the precursor;
- one peak carrying at least 98% of total intensity;
- unsorted peak arrays or duplicate peak m/z values;
- one exact-export spectrum assigned to different structures with the same formula;
- one same-CE replicate that disagrees with every peer when the remaining replicates agree;
- at least 20% orphan intensity absent from every one of at least two other CE bins;
- at least 5% intensity matching UWPR/MaConDa ions that the supplied DAG cannot explain.

The 4.5 Da precursor-envelope window prevents ordinary isotope peaks from driving the
above-precursor flag. The CSV retains the raw above-precursor fraction, the fraction inside that
window, and the corrected non-envelope fraction. A duplicate assigned to the same structure is
also retained as context but is not a review reason. Exact hashing intentionally detects only
identical exports; a tolerant near-duplicate detector would require separate validation before it
can influence exclusions.

The CSV also reports all UWPR/MaConDa mass-matched intensity as context. When multiple feasibility
files are supplied, only peaks labelled `contaminant_only` by every file increase review priority;
D3/D4 feasibility disagreements remain available as diagnostic CSV fields but are excluded from
the human-review display and priority because they describe DAG instability, not spectrum quality.

Unsupported precursor types are retained as context but do not increase review priority. The
precursor sanity check canonicalizes known NIST spellings such as `[M+H-H2O]+`,
`[M+H-2H2O]+`, `[M+CHO2]-`, and `M-H` while preserving the original label in `prec_type`.
The shared mass-shift table includes precursor-specific neutral-loss shifts. Formula-dependent
`+2i` isotope-selected precursors are excluded from this simple monoisotopic check and require an
isotope-envelope calculation. NIST's precursor m/z is normally theoretical, so a precursor
mismatch is a low-weight integrity/regression guard for label or structure inconsistency rather
than evidence of measurement mass accuracy.

## Review flag glossary

- `invalid_peaks` — **Hard:** The peak array is malformed or contains non-finite m/z values,
  non-positive m/z values, negative intensities, or no positive total intensity.
- `polarity_mismatch` — **Hard:** The precursor/adduct charge sign disagrees with the recorded
  positive or negative ion mode.
- `duplicate_spectrum_formula_conflict` — **Hard:** The same exact-export spectrum with at least
  three peaks is assigned to molecules having different molecular formulas.
- `missing_molecule_metadata` — **Hard:** The molecule ID has no usable exact molecular mass in
  the molecule table.
- `empty_spectrum` — **Hard:** The spectrum contains no peaks.
- `precursor_mass_mismatch` — **Heuristic:** Precursor m/z differs from exact molecular mass plus
  the canonicalized adduct shift beyond instrument tolerance. Because NIST precursor m/z is
  normally theoretical, this is an integrity/regression guard for label or structure
  inconsistency, not a measured-mass-quality test.
- `missing_precursor_mz` — **Heuristic:** No finite precursor m/z is available, so precursor-based
  checks cannot run.
- `substantial_intensity_above_precursor` — **Heuristic:** At least 5% of total intensity lies more
  than 4.5 Da above the precursor, outside the protected isotope-envelope window.
- `missing_collision_energy` — **Heuristic:** None of the supported absolute or normalized
  collision-energy fields is present.
- `negative_collision_energy` — **Heuristic:** At least one recorded collision-energy value is
  negative.
- `sparse_spectrum` — **Context:** The spectrum has only one or two peaks. It does not enter the
  queue or increase review priority by itself.
- `single_peak_dominates` — **Heuristic:** The spectrum has at least three peaks and one peak
  carries at least 98% of total intensity.
- `unsorted_peak_mz` — **Heuristic:** Peak m/z values are not stored in ascending order.
- `duplicate_peak_mz` — **Heuristic:** Two or more entries in the spectrum have exactly the same
  m/z value.
- `replicate_consensus_disagreement` — **Heuristic:** Among at least three same-energy replicates,
  this spectrum has similarity at most 0.50 to every peer while the remaining peers agree at 0.80
  or higher.
- `substantial_cross_ce_orphan_intensity` — **Heuristic:** At least 20% of intensity comes from
  peaks absent from every spectrum in at least two other collision-energy bins for the same
  molecule/adduct.
- `substantial_uwpr_contaminant_only_intensity` — **Heuristic:** At least 5% of intensity matches
  UWPR/MaConDa contaminant ions that every DAG source returning a feasibility decision labels
  contaminant-only; undecidable sources abstain.
- `duplicate_spectrum_structure_ambiguity` — **Heuristic:** The same exact-export spectrum with at
  least three peaks is assigned to different structures sharing one formula; this can be label
  ambiguity among isomers.
- `unsupported_precursor_type` — **Context:** The canonicalized precursor label has no entry in the
  simple monoisotopic mass-shift table, so the precursor-mass check is skipped. It does not
  increase priority and does not appear in `review_reasons`.
- `duplicate_spectrum_cross_molecule` — **Context:** The same exact-export spectrum with at least
  three peaks occurs under more than one molecule ID. The formula-conflict and
  structure-ambiguity flags provide the actionable classification; this union flag does not
  itself increase priority.
- `prec_type_was_normalized` — **Context:** The original NIST precursor label was mapped to a known
  equivalent spelling before mass and polarity checks. The original label remains in `prec_type`.
- `duplicate_spectrum_same_structure` — **Context:** The same exact-export spectrum appears under
  multiple molecule IDs having the same formula and structure key. This may reflect duplicate
  identifiers and does not itself increase priority.

## Reviewer checklist

For each candidate:

1. Confirm the NIST entry, molecule, formula, precursor adduct, polarity, and precursor m/z agree.
2. Inspect the plotted spectrum. Check malformed peaks, high-mass peaks, contaminant series,
   isotope spacing, and whether one peak overwhelms the record.
3. Compare replicate spectra and adjacent collision energies for the same molecule/adduct. A peak
   recurring smoothly across replicates is stronger evidence of real signal; an isolated peak or
   an identical spectrum attached to another molecule needs follow-up.
4. Treat DAG coverage only as supporting context. A DAG miss is not evidence by itself that a peak
   is contamination.
5. Record the decision and a short evidence-based note. Use `needs_followup` when the source entry,
   replicate evidence, or raw acquisition metadata is unavailable.

Before enabling a filter, review a random unflagged control sample alongside flagged candidates.
For consequential exclusions, use two reviewers and reconcile disagreements. Report the exclusion
rate by dataset, instrument, adduct, collision-energy range, and molecule split to detect selection
bias.
