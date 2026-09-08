# MASShatter code audit

> **Status:** REFERENCE · **Updated:** 2026-09-07 · Three issues fixed in the working tree; CPU regressions pass.

## Fix outcome and compatibility

All three findings below describe the original `0b5ad3f4` behavior. The authorized
follow-up fixes are implemented without changing configuration schemas, checkpoint
parameter shapes, or preprocessing caches:

- `MassShatterPL.forward` now constructs the same cached SMARTS templates as the
  transfer hook when standalone inference lacks them. Direct `MassShatterModel`
  callers must supply matched templates when SMARTS is enabled; missing templates
  raise instead of silently omitting mechanisms. Explicit empty matches remain valid.
- Component masses now sum neutral elemental masses, leaving the electron correction
  to the existing CRF/CMF deltas. Relative to the audited revision, each affected
  component's m/z changes by its summed precursor formal charge times 0.00054858 Da.
- SMARTS mechanism scoring now uses proposal embeddings and proposal context. Shared
  proposals use their CE-independent context. Tests cover shared/separate encoders,
  shared/per-spectrum proposals, and gradients from both objectives.

Existing checkpoints remain structurally loadable, but corrected masses and SMARTS
mechanism scores can change predictions and resumed training. Do not compare old
cached predictions with new evaluations as if the implementations were identical.
Re-evaluate affected checkpoints under one revision; pin the audited revision and
original inference path when exact historical reproduction is required. No automatic
checkpoint conversion or data regeneration is needed. Dataset-level scientific impact
has not been measured.

Validation: all MASShatter tests plus `tests/test_bond_fragment_heads.py` passed:
**718 passed, 7 skipped**, using the configured interpreter. Eleven new parameterized
cases cover the fixes and missing-template errors; the original failures were verified
before implementation. Ruff and Git whitespace checks passed. No full repository suite,
GPU validation, training, or validation-set evaluation was run. The Cython extension
was unchanged. Next action: review the diff and evaluate affected checkpoints before
using revised metrics in experiment comparisons.

## Scope and validation

Originally audited revision `0b5ad3f4`, starting with a clean working tree. Reviewed candidate
construction, model chemistry and channel selection, proposal gradient routing, Lightning
integration, and standalone inference. No model or preprocessing changes were made during
the initial audit; see the follow-up fixes above.

The configured `FRAGNNET-GPU` interpreter ran all 11 `tests/test_mass_shatter_*.py`
files: **659 passed, 5 skipped**. CUDA was unavailable; GPU behavior, the full repository
suite, training, and validation-set evaluation were not checked. The installed Cython
extension was used without rebuilding. The focused probes below reused `_build_batch`
and `_build_model` from `tests/test_mass_shatter_model.py` via `runpy.run_path`.

## 1. Standalone inference silently omits SMARTS mechanisms

[Model forward](../../src/fragnnet/model/mass_shatter_model.py) initializes empty templates
and populates them only when `static_graph.smarts_templates` exists. That sidecar is built
by [MassShatterPL.transfer_batch_to_device](../../src/fragnnet/pl_model/mass_shatter_pl.py).
However, [standalone inference](../../src/fragnnet/utils/script_utils.py), including
`process_batch_to_tabular_row`, uses `to_device` directly and bypasses that hook.
Consequently, enabling `mass_shatter_smarts_templates` changes training support but does
not enable those mechanisms in the ordinary standalone inference path.

Reproduction: evaluate `c1ccccc1S(=O)c1ccccc1` with SMARTS enabled, unique-fragment mode,
`max_masks=4`, `max_enumerated_masks=8`, precursor index 0, and NCE 25. With a populated
sidecar the model returns 9 unique components and 27 channels; without it, 7 components
and 21 channels. Six masses disappear, including 47.96643704 and 154.07770174.

Next check: route inference through the model's transfer hook or construct templates
independently of transfer, then assert identical candidate support between Lightning and
standalone inference for a SMARTS-enabled checkpoint.

## 2. Charged components receive an extra electron-mass correction

`_fragment_masses_and_valid_channels` subtracts atom formal charge times electron mass
from neutral isotope masses before applying CRF/CMF deltas. Those deltas already include
the output ion's electron correction. Components with nonzero summed formal charge
therefore disagree with the repository's formula-based mass calculation. This can also
affect fragments split from a net-neutral charge-separated precursor.

Reproduction: the intact `C[N+](C)(C)C` graph, `[M]+`, no CMF, and zero H transfer produces
**74.095877224**. The canonical
`formula_to_peak_mzs('C4H12N', '[M]+', isotopes=False)` produces **74.096425804**.
The difference is exactly one electron mass, 0.00054858 Da (about 7.4 ppm here).

Next check: establish neutral elemental component masses before applying ion deltas;
add charged-component formula parity tests. Treat a fix as a scientific behavior change
with the compatibility work required by the engineering rules.

## 3. SMARTS proposal gradients bypass encoder separation

Mechanism scoring pools `chemical_bond_embeddings` from the intensity-side
`atom_embeddings` and consumes live `context`. It does not use proposal-side embeddings
or the detached proposal context. This bypasses the gradient isolation otherwise
implemented by `proposal_selection_only` and `separate_proposal_encoder`.

Reproduction: use the SMARTS molecule and budgets above, with
`proposal_selection_only=True`, `separate_proposal_encoder=True`, and
`use_base_probability_feature=False`. Backpropagate only
`-output['proposal_fragment_logprobs'].mean()`. Absolute parameter-gradient sums are
0.29058 for `precursor_encoder`, 0.01575 for `context_encoder`, and 34.15009 for
`proposal_encoder`. The first two should remain zero under this isolation contract.
The existing isolation test covers a molecule with no SMARTS mechanisms.

Next check: derive mechanism inputs from the appropriate proposal stream and add a
SMARTS-enabled variant of the existing objective-isolation regression test.
