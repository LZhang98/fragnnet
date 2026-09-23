# Connected-fragment beam speed audit

> **Status:** FINDING · **Updated:** 2026-09-17 · Local synchronized GPU measurements;
> correctness fixes retained, no production code changes or optimizer updates.

## Result

The corrected search is substantially slower on the sampled real graphs than the earlier
small CPU probes suggested. At batch 128, beam/target 256 and boundary limit 5, GPU candidate
construction rises from **238.7 to 732.1 ms (3.07×)**. A matched forward/backward benchmark
rises from **744.4 to 1,252.3 ms (+68.2% time, 40.6% lower throughput)**. Nearly all of the
step-time increase is candidate construction; peak whole-step allocation stays near 2.83 GB.

The dominant cost is checking connected complements at every growth round, rather than
stable tie sorting. The old implementation and isolated ablations retain known correctness
defects and are timing controls only. They are not recommended alternatives to the fixes.

## Measurement contract and provenance

- Checkout: `08e128d5`, with existing unrelated working-tree changes. The corrected search
  is unchanged from that commit; baseline search source is from `dca3d0f9`. Both run against
  the same current shared candidate adapter and model. Initial dirty-diff SHA-256:
  `f2227d29a201b9ceb8e04fb71c7d788532a79172658e9eaa43b9f81965f2b278`.
- RTX 2070 SUPER, 8 GiB; Intel i7-8700K; PyTorch 2.4.1+cu121; configured `FRAGNNET-GPU`
  interpreter; two CPU threads, FP32, matmul precision `high`. GPU was initially idle.
- Resolved config: [unmerged connected-fragment arm](../../../config/massshatter/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_cmf_dual_ce_fast_gnn_proposal.yml).
  Data: `data/proc/nist20/{spec_df,mol_df}.pkl`, split `data/split/nist20_ma_mi_inchikey`.
  Seed 397623; in-memory subsample 1,024 training spectra; workers 0, shared memory off.
- Captured two batches of 256 graph instances and initialized model bond logits, then froze
  those tensors. Repeated precursors are possible because the spectra are unmerged. The first
  graph batch has atom-count quartiles 14/19/26 and maximum 76; the second has quartiles
  15/20/26.25 and maximum 95. These are sampled batch statistics, not a full-corpus census.
- Candidate cases use prefixes of the first captured batch, fixed detached logits, matching
  beam and fragment caps, and boundary limit 5. Two warmups, seven measured calls, median wall
  time with CUDA synchronization before/after each call. Variant order reverses between cases.
- Forward/backward uses one frozen real batch of 128, identically initialized models, matching
  dropout seeds per step, two warmups and five measurements. Its live logits differ from the
  separately captured candidate-only inputs. Additional synchronization brackets candidate
  construction for attribution. Data loading/transfer, optimizer updates, logging, validation
  and checkpointing are excluded, so these are not measured epoch times.
- Benchmark phases use `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Initial input capture did not set it
  and emitted a deterministic-algorithm warning; comparisons reuse the saved tensors exactly.
  Determinism is warn-only, and attention backward also warns about its algorithm. This audit
  establishes timing and finite gradients, not bitwise training reproducibility.

## GPU candidate matrix

Peak extra allocation excludes input tensors and is in decimal MB, not reserved allocator
memory. Mean fragment counts are identical before/after in these initialized-logit probes;
that does not establish atom-set equality or spectral coverage equivalence.

| Batch | Beam = target | Before ms | Fixed ms | Time ratio | Peak extra MB, before → fixed | Mean fragments |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 128 | 195.2 | 448.5 | 2.30× | 53 → 59 | 124.3 |
| 32 | 256 | 195.2 | 532.9 | 2.73× | 80 → 89 | 225.3 |
| 32 | 512 | 207.5 | 679.1 | 3.27× | 121 → 134 | 402.3 |
| 128 | 128 | 212.3 | 565.0 | 2.66× | 192 → 208 | 124.2 |
| 128 | 256 | 238.7 | 732.1 | 3.07× | 285 → 314 | 228.8 |
| 128 | 512 | 288.0 | 1,000.8 | 3.48× | 437 → 455 | 400.2 |
| 256 | 128 | 241.8 | 842.8 | 3.49× | 375 → 408 | 124.1 |
| 256 | 256 | 306.6 | 1,176.0 | 3.84× | 553 → 606 | 230.4 |
| 256 | 512 | 434.4 | 1,648.1 | 3.79× | 850 → 865 | 402.8 |

A second independent captured batch, first 128 rows, maximum 95 atoms, gives **321.1 →
1,103.2 ms (3.44×)** at beam 256; peak extra allocation is 392 → 428 MB.

CPU controls at batch 32 / beam 256 give 543.5 ms before, 720.3 ms with stable sorting alone,
1,925.9 ms with complement changes alone, and 2,089.9 ms with both fixes. Remaining large
CPU cases were stopped to prioritize GPU profiling; no paired CPU-128/256 conclusions are
drawn. One completed CPU-128 baseline remains in the raw artifact but is not a comparison.

## Isolating the fixes and whole-step impact

Both partial variants are temporary in-memory source ablations, not working-tree changes.
At batch 128 / beam 256:

| Variant | Candidate-only ms | Forward/backward ms | Candidate span within step ms | Whole-step peak MB |
| --- | --- | --- | --- | --- |
| Before either fix | 238.7 | 744.4 | 242.9 | 2,831.6 |
| Stable tie sorting only | 273.0 | 774.4 | 274.5 | 2,830.8 |
| Complement handling only | 686.0 | 1,201.5 | 704.6 | 2,833.5 |
| Both fixes | 732.1 | 1,252.3 | 749.8 | 2,830.7 |

Stable sorting adds about 34 ms to candidate construction and 30 ms to the instrumented
step. Complement handling adds about 447 ms and 457 ms respectively. The effects are not
perfectly additive. With both fixes, the candidate span is about 60% of step time, versus
33% before. Time outside the candidate span remains approximately 502 ms.

## Where the extra work occurs

An uncontended operator profile of one batch-128 / beam-256 call shows:

| Operation | Before calls | Fixed calls |
| --- | --- | --- |
| CUDA kernel launches | 24,699 | 46,417 |
| CUDA stream synchronizations | 1,175 | 3,048 |
| `aten::bmm` | 107 | 1,378 |
| `aten::equal` | 33 | 1,304 |
| `aten::nonzero` | 807 | 1,241 |
| `aten::sort` | 390 | 534 |

The [complement helper](../../../src/fragnnet/frag/mass_shatter_fast_gnn.py) runs 38 times, once
per growth round. Its profiled inclusive host span is 631 ms out of a 1,181 ms search span.
Repeated dense reachability multiplication and the host-visible equality test inside its loop
account for much of the new work. Chunking bounds workspace but does not bound total
connectivity work. Boolean indexing also incurs many synchronization-bearing operations.

Profiler instrumentation inflates runtimes; these spans describe attribution only and must
not replace the unprofiled synchronized timings above. Nested CPU/CUDA annotations are not
additive kernel time. An initial profile that briefly overlapped an audit CPU run was discarded;
the saved profile was repeated after that run stopped.

## Sensitivity to one large molecule

A controlled batch repeats captured real graphs of at most 32 atoms to make 128 graph rows.
Replacing just its final row with the captured 76-atom graph gives:

| Batch construction | Mean atoms | Max atoms | Before ms | Fixed ms |
| --- | --- | --- | --- | --- |
| All graphs at most 32 atoms | 18.88 | 32 | 97.0 | 204.4 |
| One 76-atom outlier | 19.37 | 76 | 236.2 | 582.8 |

Mean retained count is 226.9 in both cases. The fixed runtime grows 2.85× despite only a
2.6% increase in mean atom count. Dense padding, batch-maximum growth depth and connectivity
iteration depth make the longest molecule a major cost driver. This is a controlled
sensitivity test, not evidence for a specific size-bucketing throughput gain.

## Recommended next work

1. **Avoid connectivity checks for noncompetitive complements.** Compute each complement's
   own score first, then use retained valid fragments to establish a cutoff. Check potentially
   competitive complements in descending score order, preserving ties and atom-set deduplication.
   Never substitute the seed's score for its complement's score. This targets the dominant
   measured cost while retaining the correctness fix; implementation and speedup are unmeasured.
2. **Reduce reachability loop launches and host synchronization.** Benchmark a device-side
   connectivity implementation for the remaining candidates. Preserve CPU/CUDA membership and
   gradient parity, disconnected-input behavior, multiword masks and strict final budgets.
3. **Measure size-aware batching on identical total work.** The outlier experiment motivates
   it, but extra batches can add launch overhead and sampler changes affect experiments.
4. Keep stable tie behavior. If it later becomes material, benchmark deterministic partial
   selection with an explicit secondary key. It is not the first optimization target here.

## Artifacts and validation

Local artifacts are under `/tmp/connected_beam_speed_audit/` and are ephemeral:
`provenance.json` (resolved config and environment), `manifest.json` (source/input/result hashes),
`inputs_summary.json` and `full_spec_ids.json` (identities), frozen `inputs.pt`/`batches.pt`,
`candidates.json`, `sensitivity.json`, `full.json`, and `profile.json` plus profile text tables.
`audit.py` and `sensitivity.py` reproduce the measurements. Source variants are saved beside them.

From the repository root, use the configured interpreter to run `audit.py prepare`, then
`audit.py candidates`, `audit.py profile`, `sensitivity.py`, and `audit.py full` sequentially.
Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` for the benchmark phases. The candidate script includes
the larger CPU cases that were stopped in this audit; allow extra time when reproducing them.

The six previously skipped CUDA regressions now **pass**, including CPU/GPU membership and
gradient parity, staged-training integration, and tied-beam padding checks. All measured
forward/backward losses and gradients were finite. Production code, config, data and checkpoints
were not changed. No optimizer updates, cluster jobs, external logging, quality evaluation,
or full-suite rerun occurred. L40S measurements and trained-checkpoint logits remain untested.
