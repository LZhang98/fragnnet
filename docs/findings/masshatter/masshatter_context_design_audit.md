# MASShatter merged-context design audit

> **Status:** REFERENCE · **Updated:** 2026-09-08 · CE range and weighted aggregation fixes implemented; no training launched.

## Decision

Keep full metadata and conditioning in both the GNN and prediction heads as the proposed
experiment. The main pre-run concerns are the CE representation and comparison initialization,
not missing model capacity. Correct ACE clipping and arbitrary ACE selection before interpreting
the dual-CE routing comparison. A separate zero-initialized context projection is a preferable
initialization for the first routing test, but is not proven to improve held-out quality.

Scope: `918db5ac` plus the working-tree `mass_shatter_context_in_gnn` implementation and its
merged configs. Existing changes were preserved. The initial audit changed documentation only. The follow-up below implements the two
CE corrections; later numbered sections preserve the original pre-fix findings.

## Implemented follow-up: CE corrections (2026-09-08)

Both configs in the [matched merged pair](../../plans/conditioning_refactor_plan.md) now set
`ace_max: 300` and `spec_params.merge_ce_aggregation: weighted`. The global defaults and
historical experiment configs are unchanged. Duplicate proposal-sharing YAML keys were removed.

Weighted merging retains all NCE and ACE observations. Each primary-CE schedule receives equal
weight, each replicate within that schedule receives equal weight, and each reading within a
replicate shares its weight. Duplicate values are coalesced deterministically. Both MASShatter
and FraGNNet pool the individual CE embeddings with these weights; scalar conditioning and
FraGNNet CE statistics also respect them. This fixes arbitrary first-replicate selection and
stepped-schedule overweighting without changing spectral targets or merge grouping.

The new mode requires retained CE, tolerance consensus, and equal CE target weighting.
`legacy` remains the default, and the checkpoint reader explicitly supplies it for old saved
configs. Weighted batch fields are documented in the
[data reference](../../reference/dataloader_data.md). The wider ACE embedding in the new pair
requires fresh training. The GNN concatenation and its initialization are unchanged; a separate
zero-initialized projection remains an untested suggestion, not part of these fixes.

Validation:

- The broader dataset/model regression selection passed **273 tests, 5 skipped**. After adding
  two final edge/integration tests, the CE-focused selection passed **27 tests**.
- All **12,668** selected training groups have normalized weights and maximum ACE **275 eV**,
  below the new limit. Reversing rows gives identical context for all **344** affected groups.
  These checks use the source metadata before peak cleanup, not final training-loader counts.
- The corrected GNN-input config produced finite outputs and gradients with uneven CE weights
  at epochs 15 and 16 on synthetic CPU graphs. No optimizer step or quality evaluation ran.
- Changed Python files pass Ruff, except the unchanged existing `SIM110` in `proc_utils.py`;
  that file passes with this existing rule violation excluded.

CUDA was unavailable. Next check: a real-data GPU training smoke test before the matched runs.
No training or scheduler jobs were launched and no dataset artifacts were modified.

## 1. High priority: ACE silently saturates at 100 eV

Both merged configs inherit `ace_max: 100`. The
[context encoder](../../../src/fragnnet/model/mass_shatter_model.py) clamps ACE before integer
featurization, so all readings above 100 receive the same embedding input. This limit was
originally documented as a rough QTOF range; the selected data are FT/HCD.

Using the configured training split, existing scalar CE filling, and the merger's current
one-row-per-group/NCE selection:

| Quantity | Train | Validation |
| --- | ---: | ---: |
| Selected source rows | 157,678 | 52,020 |
| Merge groups | 12,668 | 4,222 |
| Retained CE readings before embedding | 152,998 | 50,621 |
| ACE readings above 100 eV | 4,843 | 1,689 |
| Groups with at least one clipped ACE | 2,714 (21.42%) | 929 (22.00%) |
| Maximum ACE after existing fill | 275 eV | 272 eV |

These counts are **before peak cleanup and any empty-spectrum removal**, not audited final
DataLoader counts. The training maximum of 275 eV also occurs in measured ACE; this is not
solely a conversion artifact. No NCE exceeds its configured 200 limit in these selected rows.

Recommended pre-run change: explicitly set `ace_max: 300` in both new comparison configs,
based on the training range, while preserving old configs. Reconstruct/retrain both arms;
the Fourier featurizer width can change when its maximum changes, so existing checkpoints
must not be silently reinterpreted. Continuous CE features are an alternative later ablation,
not necessary to resolve this saturation.

## 2. High priority: merged ACE depends on source-row order

The [merger](../../../src/fragnnet/utils/proc_utils.py) averages targets within each primary NCE
schedule, but obtains CE context with `drop_duplicates([group_id, _merge_ce_key])`. When two
replicates share NCE but have different ACE, their peaks both contribute while only the first
ACE is retained. Enabling ACE exposes this existing asymmetry.

Reproduction: two rows of one group, both NCE `[30]`, ACE `[15]` and `[25]`, identical peaks.
With `tolerance_consensus`, `equal_ce_weight=True`, and `keep_ces=True`, reversing the rows
preserves the merged peaks but changes ACE context from `[15]` to `[25]`.

In the selected training rows, after the existing ACE fill, 532 group/NCE buckets across
344 groups have ACE disagreement greater than 1 eV. Ten buckets exceed 5 eV; one exceeds
10 eV; the maximum difference is 46 eV. Thus this is not only a hypothetical input case.

Recommended pre-run change: define an order-invariant ACE aggregation within the **existing**
NCE buckets. Prefer retaining CE records and pooling their embeddings with the same hierarchical
weights as the target: equal replicate weight within a schedule, then equal schedule weight
within the merged sample. Do not fix this by grouping targets on `(NCE, ACE)` without a separate
preprocessing decision: that would change the target experiment. An explicitly chosen scalar
ACE aggregate per NCE bucket is a smaller alternative, but averaging ACE before embedding is
not equivalent to averaging its nonlinear embeddings.

This shared data path affects FraGNNet as well when merged dual-CE input is enabled. Preserve
legacy behavior behind an explicit compatibility setting and revalidate data identity when
implementing the correction.

## 3. Experiment scope: full metadata is enabled, but three fields are constant

The configured `data/split/nist20_inchikey` train, validation, and test IDs all select only:

- adduct `[M+H]+`;
- instrument `FT`;
- fragmentation mode `HCD`.

All selected rows were found, and split-file group IDs agree with the spectrum table. No groups
mix adduct, instrument, or mode. Metadata homogeneity therefore passes for this particular split.
The additional vocabulary entries in the configs do not create training examples for them.

This is a valid **FT/HCD protonated merged-CE routing experiment**, not evidence about learning
across acquisition types. Keep the five inputs, but do not infer trained support for absent
adducts/instruments/modes. Use an explicitly selected heterogeneous split if that is the target
scientific question, and audit its grouping/tolerances separately.

CE context does vary: training contains 1,869 distinct NCE schedules, with a median of 12 readings
per merged group (range 1–31). Thus constant categorical metadata does not make GNN context
injection pointless. ACE is missing in 18.24% of selected raw training rows and filled from
NCE, precursor m/z, and charge by existing code; do not describe all ACE as independently measured.

## 4. Recommended design refinement: preserve the structural initialization

The new path concatenates 128 context features to 70 atom features before a shared linear
projection to width 384. Algebraically this already computes:

```text
h0 = W_atom x + W_context c + b
```

It is valid conditioning, and the subsequent nonlinear GINE-EFA layers can learn interactions.
However, widening one `Linear` changes its initialization fan-in and consumes a different number
of random draws. Constructing both resolved models with seed 397623 gave:

| Measurement | Heads only | GNN + heads |
| --- | ---: | ---: |
| Atom input width | 70 | 198 |
| Trainable parameters | 22,701,446 | 22,750,598 |
| Atom-column initialization standard deviation | 0.0690 | 0.0409 |

Additionally, 149 of 254 common-shape trainable parameter tensors differed, including the
context MLP. Matching the seed does not pair the shared initialization across these architectures.
The 49,152 added parameters are only about 0.22% overhead; capacity cost is not the main concern.

Prefer the equivalent explicit form:

```text
h0 = atom_projection(x) + context_projection(c)
context_projection.weight = 0 initially
```

Use no extra context-projection bias. Construct the optional projection after existing modules
(or otherwise preserve the baseline RNG sequence), so shared parameters start identically.
This preserves the control's initial function and gives context an immediately trainable additive
route. It changes initialization/optimization, not the representable linear input family.
Retain direct context in the heads. This is a cleaner first comparison, not an established
spectral-quality improvement. Keep both the new implementation and its checkpoint migration
explicit if adopting it; do not silently change an already-trained concatenation checkpoint.

## 5. Deferred refinements and low-priority cleanup

- **Schedule weights:** flat mean pooling gives schedules with more stepped/ramped readings more
  context weight, whereas the target gives each schedule equal weight. A synthetic `[10]` versus
  `[20,30,40]` example confirms the mismatch. No stepped/ramped rows occur in the selected split,
  so this is not an additional blocker here. A hierarchical CE pool would address it together
  with finding 2. Permutation-invariant set encoders provide an appropriate architectural basis,
  but their general theory is not evidence of improved MASShatter accuracy.
  [Deep Sets](https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)
- **Layerwise modulation:** identity-initialized context FiLM or message gates are plausible later
  alternatives. GNN-FiLM demonstrates message modulation and reports molecular-graph results,
  but uses target-node representations for conditioning, not these acquisition metadata.
  It does not establish superiority for this task. Do not add it before fixing the CE inputs and
  testing the simple route.
  [GNN-FiLM](https://proceedings.mlr.press/v119/brockschmidt20a.html)
- **Warmup:** keep the existing 0–15 proposal-only schedule matched for this routing comparison.
  Its necessity is not established; evaluate its value separately rather than changing it in one arm.
- **Normalization:** the actual config uses BatchNorm in GINE-EFA. Cross-sample interaction during
  training through batch statistics is expected; the prior isolation test is an evaluation-mode
  property. No evidence here justifies changing normalization in the same experiment.
- **YAML:** each new merged config repeats `mass_shatter_share_proposal_across_ev: False` twice.
  Current values agree, so this is not a numerical defect. Remove the duplicate before editing
  the setting later; ordinary YAML loading silently accepts it.
- **Checkpoints:** resolved defaults retain last/best checkpoints and do not delete them;
  upload is disabled. Actual cluster storage persistence remains a launch-environment check,
  not a demonstrated failure in the conditioning implementation.

## Verification and limits

The prior focused suite passed 182 tests with 5 skips. This audit additionally constructed the
actual 384-wide GINE-EFA merged model and ran training-mode forward/backward checks on propanol
and acetic-acid graphs with multiple CE readings. At the epoch-15 phase, gradients reached the
context and GNN but not the frozen intensity head. At epoch 16, a spectral-logprob surrogate
reached all three. Predictions and gradients were finite. These checks use synthetic metadata
and surrogate losses; they are not a real-data optimizer step or a quality evaluation.

The metadata audit reads the actual spectrum table and matches split IDs, reproduces scalar
ACE filling for the selected +1 adduct, then uses the merger's current NCE deduplication.
Peak cleanup was not rerun. No datasets, model source, configs, or checkpoints were changed by
this audit; no jobs were submitted. CUDA was unavailable. No optimizer updates, real-data training
batch, full suite, throughput benchmark, or held-out model evaluation ran.

Temporary reproduction scripts/results are in `/tmp/audit_masshatter_{metadata,ce,initialization,forward}.py`
and `/tmp/masshatter_{metadata,ce,initialization,forward}_report.json`; these are local scratch,
not portable artifacts. Canonical methods, counts, and source references are recorded above.

Input provenance (SHA-256):

```text
spec_df.pkl (1,548,626,323 bytes):
997538d490a975821925b18db81af3127c832737236abdef72cb1f02e798eacd
nist20_inchikey/train_ids.csv:
16546ada839ab8aeacd330de7225995ec4b18ad62280b4b17dca8b097cc2c537
nist20_inchikey/val_ids.csv:
d86e29f4c4602a61d14c84f47355bcc2ebe52dd4c773defe5ce499ed8fa3dceb
```

Next: run a short real-data GPU smoke check of the corrected pair before the matched training
comparison. Findings 1–2 are fixed as documented above; the initialization refinement is deferred.
The [conditioning plan](../../plans/conditioning_refactor_plan.md) remains the experiment entry point.
