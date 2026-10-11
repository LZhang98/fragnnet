# Repository reorganization plan — log

> **Status:** REFERENCE · **Updated:** 2026-10-11
> Verbatim dated log moved from [the plan](../plans/repo_reorg_plan.md) on 2026-10-11, when the plan passed 600 lines; the plan holds the current summary.

Line references in these entries point at the plan as it stood when each entry was written (commits `66379ba3`, `4180d267`, `d2e3b03e`), not at the current plan.

## Progress

- 2026-10-11: plan written from a static coupling analysis at `13b1599d`. No code changed.
- 2026-10-11: revised after review, with each point checked against the code:
  - The output schema stays in core; `fragmentation/` would otherwise be imported by core.
  - Added the `_validate_tensor_condition` dependency (FraGNNet → MASSter through
    `candidate_support`).
  - Phase 2 gained a composition contract and per-stage gradient checks.
  - `misc_utils` uses `DGLGraph`; added a core batch-type registry and a `sys.modules` gate.
  - Phase 0 now exercises enabled losses, samplers across epochs and resume, export schemas, and
    every accepted model type and alias.
  - MassFormer collation follows the explicit GRAFF/ICEBERG collate chain.
- 2026-10-11: second review, each point checked against the code:
  - Phase 0 MASSter cases split: RT/CCS heads cannot be combined with proposal-only warmup.
  - Element order and `cmf_duplicate_crf_h_shift` stay in core, because core reads them.
  - A full AST check of core imports found more edges than the review named: core `feat_utils`,
    `base_dataset` (isotope cleaning reads DAG files) and `script_utils` depend on `frag_utils`,
    and `script_utils` on `masster_templates`. `frag_utils` is now split, not moved.
  - Package initializers get a lazy `__getattr__` transition and a core-only import gate.
  - Corrected: cluster jobs copy `src/` when the job starts, not at submission.
- 2026-10-11: third review, both points checked against the code:
  - `torch_scatter` is kept: PyG 2.6.1 uses it for CUDA max scatter with gradients, and
    `frag_pool_type: max` is common. The earlier "unused" claim counted only direct imports.
  - The `frag_utils` split and the `get_frag_graph`/`batch_mols_frags` extraction moved from
    Phase 6 to Phase 3, done in place. The isolation gate is now staged by phase, and Phase 6
    only renames.
- 2026-10-11: fourth review, both points checked:
  - The isolation gate is now per entry module; `script_utils` is report-only until Phase 4. A
    transitive import calculation confirmed the Phase 3 entry modules are clean. The Phase 1
    registry is now explicitly lazy for every model type, which that result depends on.
  - The registry check uses an expected-mapping table. The MassFormer dataset change in Phase 1
    is a declared change, with an equivalence check.
- 2026-10-11: user chose external ms-pred for ICEBERG. Revised the target layout, Phase 1,
  compatibility exceptions, validation and packaging accordingly. This supersedes the earlier
  optional-DGL extra and batch-type-registry proposals in the review log above. Implementation
  and environment changes have not started.
- 2026-10-11: re-read after the ICEBERG decision; fixed:
  - the stale DGL-shim sentence in the Phase 1 gate;
  - an ICEBERG pattern reference that outlived its removal;
  - the coupling-table resolution for the baselines.
  Added the concrete non-ICEBERG callers of ICEBERG code, the missing DGL install sites
  (`install_lib.py`, the Apptainer definition) and a comparability risk.
