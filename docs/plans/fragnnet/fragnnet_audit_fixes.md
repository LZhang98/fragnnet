# FraGNNet audit fixes on feiw_v2

> **Status:** REFERENCE · **Updated:** 2026-09-12 · Six fixes validated in the working tree; uncommitted.

## Objective and baseline

Fix the six FraGNNet/shared issues reproduced in the September 12 audit, on `feiw_v2`
at `bf3820b1`. The branch was already checked out and clean. It predates the shared
project-state/rules files, which were inspected read-only from revision `19a33c28`.
The [model improvement roadmap](../model_improvement_plan.md) remains the scientific
acceptance reference. MASShatter-specific changes are outside this task.

Acceptance: regression tests must reproduce the original defects, pass with the
fixes, preserve checkpoint parameter shapes and primary spectrum behavior, and
demonstrate correct probabilities, finite edge-loss gradients, and complete export.

## Changes

| Issue | Correction | Regression evidence |
| --- | --- | --- |
| Compiled checkpoints silently restore random layers | Normalize checkpoint keys using saved compilation metadata; require strict inference loading | Exact state restoration for plain, wholly compiled, and partially compiled real FraGNNet models; missing/unexpected keys raise |
| HDF5 cutoff loses the buffered final chunk | Flush on actual iterator exhaustion; apply zero cutoff consistently in the iterator | Cutoffs 0/1/2/3/10/unlimited, three chunk sizes, and inaccurate loader lengths preserve peak/node offsets and all processed spectrum IDs |
| `node_formula` explanations disagree with spectra | Derive real-state marginals from the same normalized joint probabilities as the spectrum | Formula probabilities equal normalized one-isotope spectral intensities; node-specific conditional-logit offsets leave both unchanged |
| Edge H losses crash and multiply log probabilities | Enumerate all source/destination H pairs; add their log probabilities | Explicit pair enumeration and expected loss agree, with finite gradients, for CRF-only and CRF+CMF output |
| Joint entropy accepted when its tensors are skipped | Include joint and neighbor-joint entropy consumers in the setup guard | All four raw/normalized joint entropy options reject explanation skipping; zero weights still allow it |
| NDCG/OOS metrics crash during test/labeled inference | Allocate indexed metric tensors for all splits | Test/predict values match validation over two spectra, including a fully matched spectrum |

Source: [FraGNNet model](../../../src/fragnnet/model/fragnnet_model.py),
[FraGNNetPL](../../../src/fragnnet/pl_model/fragnnet_pl.py),
[SpectrumPL](../../../src/fragnnet/pl_model/spectrum_pl.py), and
[inference utilities](../../../src/fragnnet/utils/script_utils.py).

Tests: [model/loss regressions](../../../tests/test_fragnnet_audit_fixes.py),
[checkpoint loading](../../../tests/test_script_utils_checkpoint.py),
[test metrics](../../../tests/test_spectrum_test_metrics.py), and
[HDF5 boundaries](../../../tests/test_run_inference_hdf5_node_grouping.py).

## Validation

The initial focused run reproduced 22 failures with 22 passing controls. Following
fixes and additional zero-cutoff/estimated-length coverage, all 50 focused tests pass.
The final CMF edge reference accounts for the existing suppression of unavailable
shifted CMF channels; that suppression was not changed.

An additional before/after probe loads identical weights into the original `HEAD`
model and patched model. Primary m/z, spectral log probabilities, null probability,
and OOS log probability are bit-for-bit identical for both output heads, with null
masking enabled and disabled, on `CCCO`.

Ruff passes on the model and all changed tests. The three shared production files
retain 26 pre-existing lint findings, compared against `HEAD`; no new findings.
Git whitespace checks and all nine local links in this record pass.

The initial full-suite run with the installed multi-cut `.so` had 1,963 passes,
92 failures, 14 skips, and two expected failures. All 92 failures were stale-binary
signature mismatches: 85 for `max_ring_cut_steps`, seven for `ring_open_mask`.
The temporary source-built extension from the audit was reused after verifying its
source SHA-256 exactly matches this branch's `.pyx`:
`330516d60da53a7fe6cd3f81007bc1eaf60e4a7a9136f57a2c637e723f44c3e7`.
It was loaded only in the test process; the installed binary was not replaced.

Final full-suite result with that matching extension: **2,057 passed, 14 skipped,
two expected failures**, in 53.11 seconds. Temporary logs:
`/tmp/feiw_v2_audit_before.log`, `/tmp/feiw_v2_audit_after.log`,
`/tmp/feiw_v2_audit_full.log`, and `/tmp/feiw_v2_audit_full_rebuilt.log`.
An ordinary local test run still requires a matching multi-cut extension build.

## Compatibility and next check

No config schema, checkpoint parameter shape, preprocessing, or primary spectral
intensity changes are intended. Explanation probabilities change for `node_formula`,
so re-evaluate explanations and entropy-based training comparisons under one revision.
Edge H differences now have the E×H² shape required by the existing loss consumers,
instead of the broken E×H shape. Both pair values and probabilities change correctly.
Incomplete inference checkpoints now fail instead of silently retaining random weights;
deliberate partial training transfers retain their separate API.

No packages, installed extensions, datasets, training jobs, Git staging, commits, or
branch changes are part of this work. CUDA and a real trained-checkpoint quality
evaluation remain unavailable/unperformed. Next: review the working-tree diff,
then perform target-environment scientific evaluation
before using changed optional losses for experiment comparisons.
