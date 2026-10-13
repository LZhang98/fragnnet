# Notebook audit

> **Status:** FINDING · **Updated:** 2026-10-13
> Static audit complete; NIST20 duplicate consolidated. Repair reporting errors before reusing results.

## Scope and evidence

Audited all 49 `.ipynb` files under `notebooks/` (741 code cells) at revision
`4ce6a3a4`, with pre-existing documentation and test edits in the working tree.
Used the configured FRAGNNET-GPU interpreter for JSON/schema checks, IPython-transformed
AST parsing, local import-definition inspection, Ruff undefined-name checks, and small
synthetic reproductions. Cell numbers below are one-based positions including Markdown.
This is a static audit with targeted reproductions, not an end-to-end execution certification.
No training, inference, preprocessing, dataset writes, or notebook edits were performed.

The dashboard's uncommitted isotope-module move is stale: the module is already at
`src/fragnnet/utils/isotope_cleanup_utils.py`, and current dirty files are documentation
and tests from the concurrent unit-test audit. Other sessions' edits were preserved.
See the [architecture map](../architecture.md) and
[reorganization plan](../plans/repo_reorg_plan.md) for current API ownership.

## Findings

### 1. P1 — Hungarian values overwrite binned cosine and retain binned labels

[predictions_test_analysis.ipynb](../../notebooks/exp_notesbooks/predictions_test_analysis.ipynb),
cell 9: `df["cos_sim_0.01"] = df["cos_hun"]` destroys the original binned metric.
Cell 11 then plots both columns as separate metrics, labeling the first
`Cosine Sim (tol=0.01)`; later grouped summaries inherit the replacement.
Synthetic reproduction: binned 0.2 / Hungarian 0.8 becomes 0.8 / 0.8.
This misrepresents comparisons even after the undefined metadata variable is repaired.
Keep the metric columns intact and explicitly choose the metric used by each analysis.

### 2. P2 — Bootstrap helper rescales the caller on every invocation

[calculate_bootstrap_msg.ipynb](../../notebooks/msg_notebooks/calculate_bootstrap_msg.ipynb),
cell 19: `df[metric_cols] = df[metric_cols]*100` mutates the input.
Rerunning cell 20 or 23 multiplies means and confidence bounds by another 100.
Executed the actual helper on five synthetic scores: input mean 0.3 became 30.0 after
one call and 3000.0 after two. Scale a copy inside the helper.
Its bootstrap also has no fixed random state, so identical reruns need not produce
identical intervals. Spectrum-level resampling treats rows as independent; whether a
molecule-level interval is needed depends on the intended estimand and remains unresolved.

### 3. P2 — Parameter counts labeled “nooutput” count only the output layers

[param_counts.ipynb](../../notebooks/param_counts.ipynb), cell 4:
the temporary `<model>_output` is assigned total minus output-layer parameters;
`<model>_nooutput` then subtracts this quantity from total again.
For total=1000 and output=300, it reports `nooutput=300`, although the intended
non-output count is 700. Both NEIMS and MassFormer take this path.
Use the sum of output-layer counts for `_output`, or retain the first subtraction as
`_nooutput`. Import and configuration drift currently blocks reaching this computation.

### 4. P2 — Seven notebooks import absent modules or names

These imports were checked against current defining modules, not inferred from old outputs:

| Notebook | Cell | Broken imports |
| --- | --- | --- |
| `frag_notebooks/frag_stats.ipynb` | 1 | `pl_model.FragGNNPL` |
| `msg_notebooks/calculate_bootstrap_msg.ipynb` | 1 | `loss.sparse_cosine_distance`, `loss.sparse_jensen_shannon_divergence` |
| `param_counts.ipynb` | 2 | `pl_model.FragGNNPL`, `pl_model.NeimsPL`, absent `iceberg.pl_model` |
| `test_notebooks/test_bin_sim.ipynb` | 1 | `spec_utils.calculate_sparse_cosine_similarity` |
| `test_notebooks/test_formula_pkls.ipynb` | 8 | `spec_utils.calculate_cosine_similarity` |
| `test_notebooks/test_graff_ckpts.ipynb` | 2 | `inference.runners.run_inference`, `spec_utils.calculate_sparse_cosine_similarity` |
| `test_notebooks/test_ints_transform.ipynb` | 2 | `spec_utils.batched_l2_normalize`, `misc_utils.safelog`, `misc_utils.scatter_reduce` |

Retarget supported names to their defining modules and adapt removed signatures against
current code. ICEBERG requires the external integration or an explicitly historical notebook;
do not restore retired APIs solely to run these notebooks.

### 5. P2 — Clean execution fails on syntax and undefined variables

- [debug_nist20_uwpr_contamination_analysis.ipynb](../../notebooks/debug_notebooks/debug_nist20_uwpr_contamination_analysis.ipynb),
  cell 41, lines 77–78 and 85–86: ordinary quoted strings contain literal line breaks;
  AST parsing and Ruff both reject the cell.
- [predictions_test_analysis.ipynb](../../notebooks/exp_notesbooks/predictions_test_analysis.ipynb),
  cell 9: `spec_meta_cols` is never defined.
- [ablation_exp.ipynb](../../notebooks/exp_notesbooks/ablation_exp.ipynb),
  cell 5: export uses undefined `sims_latex_df`; earlier cells create `latex_df`.
- [NIST20 supplemental fragment analysis](../../notebooks/data_notebooks/nist20.ipynb),
  cells 18–19: `spec_frag_stats_df` is referenced after its initialization was commented out.
- [test_ints_transform.ipynb](../../notebooks/test_notebooks/test_ints_transform.ipynb),
  cells 7–9: `norm` is never assigned (only `norm1` and `norm2` exist).

Stored execution counts and outputs do not prove a fresh kernel can run these notebooks.

### 6. P2 — Condition plots hard-code adduct IDs and mislabel NCE units

[ikohsc16_10ppm.ipynb](../../notebooks/condtion_notebooks/ikohsc16_10ppm.ipynb),
cells 9–10: the hard-coded adduct mapping labels index 1 `[M+Na]+` and index 2 `[M-H]-`.
Current `BaseDataset._setup_prec_type_to_idx` assigns indices by sorted configured names;
for these three adducts index 1 is `[M-H]-` and index 2 is `[M+Na]+`.
The instrument mapping also assumes exactly two indices; current configurations can
include ORB, ORB_F and QTOF. Decode from the originating configuration or join string
metadata rather than infer identities from integer codes.
Cell 5 selects QTOF ACE and otherwise NCE, while cell 9 labels all x axes as eV.
NCE panels need an NCE label. Verify the source export's actual index mapping before
using existing figures; its configuration was not recovered in this audit.

## Additional reproducibility limitations

- Multiple notebooks use `../data` from nested notebook directories, whereas others use
  `../../data` or find the repository root. Without a documented working directory,
  default execution from each notebook's directory resolves different stores.
  `test_ann_df` and `test_graff_ckpts` also change into another user's absolute checkout.
- Saved errors include missing historical retrieval/fragment files, an interrupted MSGYM
  cell, a missing isotope annotation field, and empty-array plotting failure. These are
  historical outputs, not reproduced failures against current data.
- `nbformat.read(..., as_version=4)` followed by validation rejected four notebooks:
  `debug_nist20_uwpr_contamination_analysis`, `debug_nits20_peaks`,
  `predictions_frag_explainability`, and `nist20_frag_gen_comparison`; cell `id` fields
  conflict with their declared older schema. Other notebooks emitted missing-ID warnings.
  Normalize schema metadata in a separate repair pass; files were not rewritten here.
- The random retrieval baseline in `ms2c_retrieval_exp` uses `len(ks)` as candidate count.
  Its current `ks=1..50` matches the named 50-candidate setup, so this is not a demonstrated
  current error. Varying the plotted ranks or candidate-set sizes would invalidate it.

## Validation and handoff

All notebook JSON parsed; 740 of 741 code cells parsed after IPython transformation.
Ruff `F821,F822,F823` independently confirmed the syntax/undefined-name failures.
The directory-wide Ruff command also inspected the existing scratch
`test_notebooks/test_fraggnn_formula.py`; its undefined names are outside the notebook count.
Synthetic checks confirmed findings 1–3 without reading or writing experiment artifacts.
No full notebook executions, GPU checks, or model tests were run.

Next: repair findings 1–3, then imports and fresh-kernel blockers; recover originating
configurations for condition plots. Run selected repaired notebooks from an explicit working
directory with known inputs and review metric definitions before accepting new figures.

## Duplicate consolidation (2026-10-13)

Compared all 47 notebooks present at the start of this pass by code and Markdown source,
ignoring execution metadata and outputs. No exact source duplicates were found.
`notebooks/frag_notebooks/nist20_frags_stats.ipynb` repeated 17 cells from
[the retained NIST20 notebook](../../notebooks/data_notebooks/nist20.ipynb).
Moved its four nonempty unique cells, including their saved outputs, to the end of the
retained notebook, then removed the redundant file. The retained notebook's original
18 cells are unchanged. Every nonempty source cell from both notebooks is preserved;
its supplemental cells still have the pre-existing undefined `spec_frag_stats_df` issue
(finding 5), now at cells 20–21. No scientific calculation was repaired in this pass.

The Spectraverse/MSGYM pair uses different datasets; the baseline/MBFS and MBFS/prepass
pair compares different DAG mechanisms. Both pairs were retained. Existing condition-
notebook deletions and the isotope-notebook rename were left untouched. Source-preservation
checks and JSON parsing passed; no notebooks or model tests were executed.
