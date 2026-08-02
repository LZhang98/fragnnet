# NIST suspicious-spectrum human review

Generate a ranked queue and visual report:

```bash
python scripts/eval/nist_bad_data_review.py \
  --spec_fp data/proc/nist20/spec_df.pkl \
  --mol_fp data/proc/nist20/mol_df.pkl \
  --uwpr_feasibility_csv \
    data/uwpr/nist20_uwpr_hits_feasibility.csv \
    data/uwpr/nist20_uwpr_hits_feasibility_d4.csv \
  --out_csv data/eval/nist20_bad_data_review.csv \
  --out_html data/eval/nist20_bad_data_review.html
```

The priority score orders work. It is not a probability and must not be used directly as a
training filter. The CSV preserves separate hard and heuristic reasons and includes blank reviewer
fields. Record one of `keep`, `exclude`, or `needs_followup` in `review_status`; never edit the raw
NIST tables during review.

## Signals

Higher-confidence consistency checks:

- malformed, non-finite, negative-intensity, or zero-total-intensity peak arrays;
- precursor m/z inconsistent with molecular exact mass and the assigned adduct;
- missing precursor m/z;
- precursor-adduct polarity inconsistent with `ion_mode`;
- the same normalized, high-resolution spectrum assigned to different molecules;
- missing molecule metadata.

Review heuristics, not proof of bad data:

- no collision-energy metadata or negative collision energy;
- at most two peaks;
- at least 5% of intensity above the precursor tolerance;
- one peak carrying at least 98% of total intensity;
- unsorted peak arrays or duplicate peak m/z values;
- one same-CE replicate that disagrees with every peer when the remaining replicates agree;
- at least 20% orphan intensity absent from every one of at least two other CE bins;
- at least 5% intensity matching UWPR/MaConDa ions that the supplied DAG cannot explain.

The CSV also reports all UWPR/MaConDa mass-matched intensity as context. When multiple feasibility
files are supplied, only peaks labelled `contaminant_only` by every file increase review priority;
DAG disagreements are reported separately. These labels depend on the DAG, so record each
depth/configuration with the review artifact.

Unsupported precursor types are retained as context but do not increase review priority. The
precursor sanity check uses precursor-specific shifts for neutral-loss labels; the fragment-ion
carrier mapping must not be used directly for those labels. Formula-dependent `+2i`
isotope-selected precursors are excluded from this simple monoisotopic check and require an
isotope-envelope calculation.

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
