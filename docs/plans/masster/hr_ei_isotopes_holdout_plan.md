# Exact-mass EI isotopes and an HR-EI-excluded nominal training split

> **Status:** PREPARED · **Updated:** 2026-10-17
> Exact-mass natural isotope output implemented; HR-EI-excluded nominal split and fresh-training configuration prepared. No training submission.

## Objective and acceptance

Support natural isotope envelopes in MASSter exact-mass output, conserving channel
intensity and retaining nominal behavior/checkpoint compatibility. Isotope masses
must use elemental isotope mass differences, including negative offsets and fine
structure. Validate chemical examples, intensity conservation, backward gradients,
and the frozen EI checkpoint. Plain spectrum exports must work; explanatory exports
must retain an explicit guard if isotope labels cannot be represented safely.

Prepare a new split with no connectivity overlap with resolvable HR-EI reference
identities in training, validation, or internal test. Preserve original split
assignment for retained rows and leave original files unchanged. Record input hashes,
excluded identities, removed rows, and zero-overlap checks; prepare a fresh-training
configuration using the new split. Do not initialize it from the contaminated e1 checkpoint.

## Established evidence

- [Raw audit](../../findings/masster/masster_ei_evaluation.md#hr-ei-raw-peak-and-structure-audit): four libraries;
  Castro structure mapping remains unresolved.
- [Frozen e1 comparison](../../findings/masster/masster_ei_evaluation.md#frozen-masster-ei-checkpoint-on-measured-hr-ei-spectra):
  962 consistent mapping records evaluated, but exact output currently lacks isotopes.
- Existing natural isotope rows require nominal mass mode. Their nominal mass offsets
  cannot be reused as exact-mass increments.

## Scope and policy

Use fixed natural abundances with the existing maximum nominal isotope order and
minimum-share settings. Exact isotopologues are resolved by isotope mass; nominal
settings keep their current semantics. Exclusions conservatively include valid
computed and supplied raw-audit identifiers plus native parent structures supplied
in RECETOX SDFs. Unknown Castro identities are not certified excluded and must not
enter a claimed held-out benchmark until mapped.

## Progress

Implemented in
[`masster_exact_isotopes.py`](../../../src/fragnnet/models/masster/masster_exact_isotopes.py)
and the MASSter model's isotope expansion. Fine structure uses natural elemental
isotope distributions and 1e-8 Da integer mass-increment keys to combine atom
permutations without floating-point ordering artifacts. Only states that can reach
the final signed nominal window are retained during convolution. Probability
thresholding occurs after full convolution/window normalization, followed by
renormalization of retained fine-structure rows. Ion compositions include adduct
atoms and the channel-specific hydrogen shift. Scores remain differentiable; no
new learned state is added. Exact output masses are float64, and pathway annotations
include `pred_pathway_isotope_mass_shifts` alongside the existing nominal offsets.
Nominal output retains its prior algorithm. Plain spectrum exports work;
explanatory tabular isotope exports retain their explicit unsupported guard.

Prepared split:
`data/split/nist23_ei_inchikey_hr_ei_excluded_2026_10_17/`.

| Partition | Original spectra | Retained spectra | Removed spectra |
| --- | ---: | ---: | ---: |
| Train | 154,981 | 154,206 | 775 |
| Validation | 51,700 | 51,442 | 258 |
| Internal test | 51,685 | 51,451 | 234 |
| Secondary | 0 | 0 | 0 |

There are 990 distinct exclusion connectivity keys. Retained partitions have zero
overlap with those keys; original ordering and assignment are preserved. The split
contains `excluded_identities.csv` (source, identity field, name and record),
`removed_rows.csv` (original IDs and partition), and `meta.json` (policy, source
input SHA-256 hashes, original metadata, counts and verification). Exclusions use
the raw audit's valid computed/supplied keys and both RECETOX SDF inventories.
The original processed molecule/spectrum tables are shared read-only; the split
controls which rows a new run can load.

The source molecule table has 253,755 rows, as recorded in the raw audit; this is a
split of the current local data, not a reconstruction of remote e1 training inputs.
Unknown Castro identities remain unresolved. The 962 mapped reference spectra are
covered, but no claim is made that unmapped external records are excluded.

Reproduce into a fresh directory with
[`prepare_hr_ei_holdout_split.py`](../../../scripts/debug/ei/prepare_hr_ei_holdout_split.py):

```bash
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python -m scripts.debug.ei.prepare_hr_ei_holdout_split \
  --output data/split/nist23_ei_inchikey_hr_ei_excluded_2026_10_17
```

The directory already exists; do not rerun into it. The script refuses overwrites.
Fresh-training recipe:
[`e5_hr_ei_holdout_nominal.yml`](../../../config/masster/ei/e5_hr_ei_holdout_nominal.yml).
It retains e1's nominal targets, natural isotope training, seed and architecture,
uses the new split and a distinct run/group name, and explicitly sets
`finetune_pretrained_ckpt: none`. No training job has been launched. The existing e1
checkpoint remains contaminated relative to this reference set and is only an
implementation-validation baseline.

The inference overlay
[`inference_hr_ei_natural.yml`](../../../config/masster/ei/inference_hr_ei_natural.yml)
selects exact-mass natural isotope output while preserving the checkpoint's other
settings. The EI method reference describes the output semantics and annotation limit.

Validation includes exact C/N M+1 fine structure and chlorine mass increments,
fine-to-nominal convolution agreement including selenium/bromine cancellation,
three precursor types, score/intensity conservation, backward gradients, plain
export fidelity, annotation rejection, and a real sparse Gaussian training loss.
The nominal isotope regression suite passes. GPU cases are skipped because CUDA
is unavailable locally. Final focused validation: 45 tests passed, four GPU-only
cases skipped; Ruff and diff checks pass. The frozen e1 checkpoint loads strictly
with exact natural output; all 962 reference spectra completed on CPU. The
[follow-up comparison](../../findings/masster/masster_ei_evaluation.md#follow-up-natural-exact-mass-isotope-output)
records results, runtime, and artifacts. They remain a contaminated baseline.

Original artifacts and unrelated UMAP/CASMI edits remain untouched.

## Next action

Train e5 from scratch on the prepared split, then evaluate its exact natural isotope
output on the fixed HR-EI reference set. Do not fine-tune e1 and call that held out.
Resolve Castro structures separately before adding that source to the benchmark.
