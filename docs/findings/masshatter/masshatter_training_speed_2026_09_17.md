# MASShatter training speed: s3gy1ro4

> **Status:** FINDING · **Updated:** 2026-09-18 · Reduced monitoring, CPU improvements,
> optional exact CUDA construction and a batched proposal coverage loss measured locally;
> target L40S acceptance remains pending. No cluster job or remote run was changed.

## Result and implementation

The initial measured improvement was to disable **monitoring-only** channel attribution and
base-spectrum CE. The [fast config](../../../config/massshatter/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_cmf_dual_ce_fast.yml)
differs from the [s3gy1ro4 control](../../../config/massshatter/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_unique256_enum4096_gine_gsa384_bondcentric_a2_b3_transformer_full_context_dual_ce_proposal_selection_only_inst_tol.yml)
in exactly four resolved keys: W&B name/tags and these existing switches:

```yaml
mass_shatter_channel_diagnostics: False
mass_shatter_log_base_ce: False
```

Candidate budgets, chemistry, metadata, spectral/proposal objectives, precision, batch sizes,
gradient accumulation, optimizer, and seed are unchanged. The control remains available.
These switches remove the two diagnostic families in **both training and validation**.
Validation spectral loss, cosine, candidate coverage, and inexpensive candidate/channel counts
remain available. Monitoring can be restored for a diagnostic evaluation.

No production Python change was needed for that monitoring-only step. Strict checkpoint
state-dict loading between those two configs succeeds, and seeded initial states are identical.
The subsequent CPU and optional CUDA implementations and their measurements are recorded below.

## Live W&B evidence

Read [s3gy1ro4](https://wandb.ai/frag-gnn/frag-gnn/runs/s3gy1ro4) through the authenticated,
read-only API. The snapshot contains all 18 history rows (train/validation epochs 0–8) and
3,493 system-history rows, of which 1,747 contain GPU utilization. W&B reports `running`;
the latest complete epoch is 8, validation cosine 0.693215, after approximately 6.69 hours.
System samples extend to approximately 7.28 hours. This is a snapshot, not a completion report.

| Observation | Value |
| --- | ---: |
| GPU | One NVIDIA L40S |
| Median epoch interval, epochs 1–8 | 2,663 s / 44.4 min |
| Optimizer steps per epoch | 1,124 |
| Training microbatch / accumulation | 128 / 2 |
| Precision / deterministic algorithms | FP32 / disabled |
| Median sampled GPU utilization | 33% |
| Median sampled allocated GPU memory | 5.22 GB / 10.8% |
| Median sampled GPU power | 118.5 W of a 350 W limit |

Epoch intervals include validation and epoch-end work. They are not pure training-step timings.
The low utilization and memory allocation support looking at host work and synchronization;
they do not identify a single bottleneck by themselves. The historical config's `deterministic`
tag does not mean that `deterministic: True` was active.

Epoch-8 training candidate timings, averaged over batches:

| Phase | ms per batch |
| --- | ---: |
| Input transfer | 0.03 |
| Mask selection | 71.06 |
| Topology | 166.91 |
| Local assembly | 40.17 |
| Output packing **plus** transfer | 83.81 |
| Total candidate construction | 416.06 |

The output-transfer metric includes packing; it is not a measurement of PCIe transfer alone.
The listed phases do not exhaust the total. These are the run's internal host timers, not a
synchronized CUDA profile. The model reports about 11,097 component pathways per molecule but
only 232 unique fragments; about 95.5% of component occurrences repeat a unique fragment.
Removing those latent pathways would alter proposal marginalization unless equivalence is proven.

## Why monitoring costs so much

In [mass_shatter_pl.py](../../../src/fragnnet/pl_model/mass_shatter_pl.py), the loss closure computes
channel attribution, support metrics and required-top-k quantiles whenever channel diagnostics
are requested. This happens even with `compute_train_auxiliary_metrics: False`, because these
metrics are registered as loss metrics. In this run both channel-support loss weights are zero:
the entire attribution calculation is for monitoring.

[candidate_support.py](../../../src/fragnnet/model/candidate_support.py) loops over spectra, repeatedly
compacts GPU tensors, matches dense channels to observed peaks, and reads GPU scalar conditions.
Required-top-k diagnostics add per-spectrum masking and quantiles. The baseline also evaluates
Gaussian CE a second time for base-spectrum logging despite `mass_shatter_base_ce_weight: 0`.
These calculations are independent of the nonzero spectral and proposal objectives.

Candidate generation remains a separate major cost: proposal GPU work is followed by CPU mask
selection/topology/assembly and then candidate upload before the intensity head can continue.
Increasing DataLoader workers does not parallelize that forward-pass topology loop. Existing
early termination, compiled selection and packed assembly are already present; this audit did
not treat them as missing optimizations.

## Matched local measurements

Local checkout started at `a4c87b6461bc40ad66b86acdcbfd8cc3888fff3e` with pre-existing
documentation edits. W&B does not supply a Git revision for this run, so remote/local source
identity is unverified. Measurements use the configured `FRAGNNET-GPU` interpreter and an
RTX 2070 SUPER (8 GB), not the run's L40S.
The recorded run config matches the current resolved control except for the local template's
additional empty `spec_params.split_filters` mapping; no active filter differs.

The final [benchmark](../../../scripts/debug/benchmark_mass_shatter_training.py) loads a seeded
1,024-spectrum sample of the same NIST20 MA/MI training split, prepares four fixed batches of
128, and repeats each configuration from identical seeded initialization and dropout seeds.
It uses two PyTorch CPU threads, FP32, four warmup batches and twelve measured batches. CUDA is
synchronized at measurement boundaries. The timed region includes spectrum preprocessing,
model forward, losses/metrics and backward. It excludes data loading, batch transfer, optimizer
updates, accumulation orchestration, validation and W&B logging. No model is trained to convergence
and no run checkpoint is used.

| Final repeat | Median forward/backward | Peak PyTorch allocated memory |
| --- | ---: | ---: |
| Original monitoring | 1,235.8 ms | 3.98 GB |
| Both monitoring switches off | 885.4 ms | 3.74 GB |

This is **28.4% less batch time / 1.40× forward-backward throughput**. Candidate construction
was nearly unchanged (477.9 versus 475.5 ms). An earlier batch-128 repeat measured 1,283.0
versus 886.2 ms (1.45×); a batch-32 diagnostic measured 338.6 versus 235.2 ms (1.44×).
At batch 32, disabling channel diagnostics alone measured 246.5 ms and disabling base CE alone
318.4 ms, identifying channel diagnostics as the larger removable expense.

These are local diagnostic measurements, **not a demonstrated 1.4× L40S epoch speedup**.
GPU full-forward loss differences reached 0.000524 across repeated configurations with
nondeterministic kernels and learned discrete candidate selection. Synthetic fixed-input
regressions verify exact equality of the optimized loss and spectral/proposal gradients for
each monitoring-switch combination. A production run should compare resolved flags and actual
epoch throughput; no claim of bitwise-identical training trajectories is made.

An attention-layout experiment replaced histogram counting and per-bucket compaction with
sorted ranges. CPU/CUDA forward, packing and gradient equivalence passed, but the isolated
head gain was only 73.36 → 72.00 ms on a synthetic ragged workload. With monitoring off, full
batches measured 886.2 → 891.0 ms. The implementation and its experiment-specific tests were
removed because a useful whole-batch improvement was not established.

Local diagnostic artifacts (temporary, not portable): `/tmp/masshatter_s3gy1ro4_audit.json`,
`/tmp/msh_training_benchmark.json`, `/tmp/msh_training_benchmark128.json`,
`/tmp/msh_monitoring_final.json`, and `/tmp/msh_head_benchmark.json`. The final benchmark JSON
records the resolved config, sampled spectrum IDs, device, PyTorch version and individual timings.

## Reproduce and use

From the repository root, with the configured environment:

```bash
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python \
  scripts/debug/benchmark_mass_shatter_training.py \
  --config config/massshatter/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_unique256_enum4096_gine_gsa384_bondcentric_a2_b3_transformer_full_context_dual_ce_proposal_selection_only_inst_tol.yml \
  --output /tmp/masshatter_monitoring_benchmark.json
```

The benchmark supports CPU fallback and `--batch-size` for smaller devices. For a new production
job use the fast config through the [cluster submission procedure](../../guides/submit_cluster_runs.md),
with a new job ID. To preserve the current run's training progress, use a controlled checkpoint
resume instead: inspect the resolved config **after W&B initialization** before continuing, because
[runner.py](../../../src/fragnnet/runner.py) copies W&B config values back into the local config.
A changed local YAML or CLI override alone is not proof that a resumed run uses the new flags.
The remote checkpoint and resumed behavior were not tested or modified here.

Validation: 202 focused model/PL/context tests passed, five CUDA tests skipped in that CPU run;
three of the passing tests check exact monitoring-switch loss/gradient equivalence. Config
resolution and strict state-dict compatibility passed. The final real-data GPU benchmark completed.
Ruff and whitespace/link checks passed. Full-suite testing and validation-set quality evaluation
were not run; production model code and scientific objectives are unchanged.

## Larger-microbatch follow-up (2026-09-17)

At the user's request, the [batch-256 config](../../../config/massshatter/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_cmf_dual_ce_fast_batch256.yml)
changes `train_batch_size: 128 → 256` and `accumulate_grad_batches: 2 → 1` relative to the fast
config. Only those two settings and W&B name/tags differ. Effective batch size remains 256 on
one GPU; LR, validation batch size, losses, precision and candidate budgets stay fixed.
Initialization is identical and checkpoint tensor shapes are unchanged. The model contains four
BatchNorm1d modules, so larger microbatches change normalization statistics; dropout and candidate
selection can also differ. Equal effective batch size does not imply identical training updates.

The local GPU completed both batch-256 forward/backward repetitions with finite losses and no
OOM. Both monitoring switches were already off, so the benchmark's two variants act as repeats
here. The controls used exactly the same ordered 512 spectrum IDs from the same 1,024-spectrum
subsample: four batches at 128 versus two at 256. Each repetition warmed up on 512 spectra and
timed 1,024 spectra. The earlier benchmark limitations still apply, including fresh weights,
no optimizer updates, two CPU threads and no end-to-end DataLoader/validation timing.

| Microbatch | Spectra/s, repeat 1 / 2 | Candidate ms/batch, repeat 1 / 2 | Peak allocated memory |
| --- | ---: | ---: | ---: |
| 128 | 145.1 / 144.4 | 471.9 / 473.7 | 3.73 GB |
| 256 | 140.6 / 143.9 | 974.0 / 940.6 | 7.10 GB |

The larger batch fits this sampled workload but does **not establish a throughput gain on the
RTX 2070 SUPER**. Candidate time grows approximately linearly; increased VRAM use is not evidence
of increased GPU utilization or faster training. No new GPU-utilization sampling was collected.
The L40S result remains unmeasured, and these short samples do not prove every batch fits.

Artifacts: `/tmp/msh_batch128_control.json` and `/tmp/msh_batch256.json`. Reproduce on an allocated
L40S from the repository root, with that job's Python environment:

```bash
python scripts/debug/benchmark_mass_shatter_training.py \
  --config config/massshatter/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_cmf_dual_ce_fast.yml \
  --batches 4 --warmup 4 --steps 8 --output /tmp/msh_batch128_control.json
python scripts/debug/benchmark_mass_shatter_training.py \
  --config config/massshatter/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_cmf_dual_ce_fast_batch256.yml \
  --batches 2 --warmup 2 --steps 4 --output /tmp/msh_batch256.json
```

The benchmark now reports spectra/second from actual timed batch sizes, alongside latency and
memory, to make comparisons across microbatch sizes explicit. Config-difference/effective-batch
checks and identical initialization passed; no production model code changed and no cluster run
was launched. No convergence or validation-quality comparison was performed.

Next: compare 128×2 and 256×1 on L40S using spectra/second, epoch time, peak memory and sampled GPU
utilization, with the same monitoring settings. If testing 512×1 afterward, record that effective
batch size doubles to 512; its optimization comparison is separate. Profile the remaining topology
and packing under the [existing performance gates](../../plans/masshatter/masshatter_topology_assembly_performance_plan.md)
to address the CPU interval between GPU stages. Mixed precision and smaller enumeration budgets
remain separate experiments.

## Batch-512 config prepared (2026-09-17)

The requested [batch-512 config](../../../config/massshatter/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_cmf_dual_ce_fast_batch512.yml)
differs from the batch-256 variant only in `train_batch_size` and W&B name/tags.
Accumulation remains 1, so effective batch size is 512 on one GPU. LR remains 0.001;
validation batch size remains 64, precision remains FP32, and both monitoring switches stay off.
Resolved-config checks passed at preparation; no training run was launched then. The subsequent
CPU-phase local benchmark hit OOM at 512 on an 8 GB GPU; the L40S memory/throughput check and
convergence comparison remain pending.

## Candidate-sampling follow-up (2026-09-17, source audit)

The subsequent [implementation plan](../../plans/masshatter/masshatter_candidate_backend_plan.md) records the
user-selected sequence: CPU improvements first, then a separately selectable GPU backend.
At the time of this source audit the opportunities below were unimplemented. The later
[implementation measurements](#cpu-implementation-and-full-training-measurements-2026-09-17) record
which changes were retained, removed or deferred.

The user asked how to speed up sampling. This refers to ranked bond-cut candidate construction;
the current path enumerates masks deterministically rather than drawing independent random masks.
Source inspection confirms four selection threads, a Cython `nogil` enumeration kernel,
single-mixture sort/dedup bypasses, early topology termination and packed host assembly already
exist. These are not new optimization opportunities.

The remaining priorities are:

1. **Small selector changes:** in `mass_shatter_core.pyx`, `top_mixture_bond_sets_raw` still
   recomputes every mask probability for one mixture with both probability floors disabled.
   Those scores only feed the floor check on this path; differentiable probabilities are computed
   separately in the candidate builder. A specialized no-floor branch could avoid that work,
   subject to extreme-logit/numerical-contract tests. The two numeric CSR-output loops also run
   outside `with nogil`; separating their allocations from their numeric loops would let more
   of the existing threaded selection overlap. Neither change has been implemented or timed.
2. **Parallel topology/local assembly:** the independent per-molecule construction loop and
   `build_component_arrays_raw` still execute serially with the GIL. Split allocation/validation
   from numeric `nogil` work, then use bounded native threads and deterministic result ordering,
   under Stage 2's existing correctness and throughput gates. This targets the measured 167 ms
   topology stage plus local assembly, a larger pool than selection's 71 ms.
3. **Stream selection into topology:** the selector still materializes the full ranked budget
   before topology stops at the unique-fragment target. A resumable heap plus incremental topology
   could avoid the unused suffix. Preserve forced-mask order, tie order and the existing strict
   overshoot proof; stopping immediately at equality with U256 is not equivalent. This is a larger
   change, and spectra that exhaust their budget cannot benefit from early termination.
4. **Compact memberships before transfer:** unique mode uploads occurrence-level atom/boundary
   memberships and then discards duplicates in `_unique_component_memberships`. Retaining these
   memberships once per representative could reduce packing/transfer and GPU compaction. Preserve
   all latent pathway metadata used in probability marginalization; do not delete duplicate
   pathways merely because their atom sets match. This remains Stage 4 of the existing plan.

A smaller `mass_shatter_ranked_mask_budget` (4096 → 2048 first) is a separate speed/coverage
experiment, not an equivalent optimization. Hold the unique target at 256 and compare candidate
weighted recall, validation cosine and spectra/second. Existing historical E1024/E4096 comparisons
show a small but nonzero coverage/cosine gain at E4096; they do not establish the cost of lowering
the current CMF run's budget. More DataLoader workers do not parallelize candidate construction
inside the model forward. No sampler implementation or configuration was changed in this audit.

## CPU implementation and full-training measurements (2026-09-17)

The CPU-first plan is now being implemented. The retained local change releases the GIL around
numeric topology/assembly and selector CSR loops, shares one persistent process-local pool across
selection and topology, and assembles group representatives in deterministic input order. Large
raw unique-array views are compacted; private local buffers are consumed on their final group view
to avoid duplicate remapping allocations. CPU remains the default. Kernel source was rebuilt
before validation; no scheduler job or W&B training run was created.

The new [full-training harness](../../../scripts/debug/benchmark_mass_shatter_full_training.py)
includes production data loading, device transfer, loss/backward, accumulation, gradient clipping,
AdamW updates and step scheduling when configured. It excludes W&B/Lightning logging, validation
and checkpoints. [L40S instructions](../../guides/benchmark_masshatter_cpu.md) freeze code and rebuild
native kernels on the target; all suggested allocations remain **three hours**.

Local environment: RTX 2070 SUPER 8 GB, PyTorch 2.4.1+cu121, FP32, seed 397623, six DataLoader
workers, two Torch CPU threads, four candidate threads, fast config with U256/E4096. Measurements
start from seeded weights, **not the mature `s3gy1ro4` checkpoint**. Both variants have state SHA256
`f6138e2da037139ad598d6200563cb8bf2a10c6f7845ae2aa56b1a1f6dd06570` and identical ordered spectrum IDs.
The baseline uses the original Python/native pair from `a237afe79a24df1347918653e757f43306dcefb8`.
Complete resolved settings, source/binary hashes, input file path/size/mtime, IDs and commands are
in the [local artifacts](../../../output/masshatter_cpu_2026_09_17/); dataset contents were not hashed.

### Stage decisions

Each row below measured 12,288 spectra: two warmup optimizer updates, then three windows of
16 updates, batch128×accumulation2, 4,096 sampled spectra and continuous optimizer state.

| Variant | Spectra/s | Relative to original CPU | Decision |
| --- | ---: | ---: | --- |
| Original CPU | 219.93 | — | Reference |
| A1 score/buffer specialization plus CSR GIL release | 226.65 | +3.06% | Score/buffer specialization removed: misses 5% gate |
| A1 plus initial parallel topology | 239.32 | +8.82% | Parallel topology improves 5.59% over A1 |
| Final CPU: parallel topology, CSR GIL release, compact buffers/remap; original scoring | 240.29 | **+9.26%** | Retained provisionally; L40S gate remains open |

The final row includes lightweight membership profiling (two CUDA events per forward, no added
per-batch synchronization). The matched uninstrumented thread sweep below reaches 239.54 spectra/s
on a shorter window. These are local diagnostics: the aggregate 10% target has **not** been met,
and the canonical L40S/three-hour validation and checkpoint-inclusive gates remain open. CUDA
reductions are nondeterministic in this configuration, so loss and candidate-count trajectories
vary slightly despite matching initial weights and input IDs; fixed-input discrete parity is tested
separately. Do not interpret these windows as a convergence or validation-quality comparison.

A1's isolated selector benchmark improved 1.67×, but this did not justify keeping its extra
score-free branch. The reference extreme-FP64 behavior and independent selector tests remain.
A3's upload-only membership compaction is deferred: the final profile uploads a mean 109.18 MB of
candidate indices per microbatch, of which occurrence memberships occupy 64.76 MB and the compact
replacement would occupy 7.77 MB. Remap/packing plus transfer takes 37.85 ms; GPU membership filtering
spans 0.84 ms. Optimistic byte-proportional savings estimate about **4.0%** full-step improvement,
before paying for CPU compaction. Removing native membership materialization as well would require
a broader contract/kernel change and a new target-device profile.

A4 is deferred: mean selected/scanned counts are 483,652/154,939 per microbatch. Even treating
selector cost as proportional to the unused suffix projects only about **4.4%** full-step gain;
heap/validation overhead and the full-pool single-cut diagnostic make that optimistic. Neither
stage changes the enumeration budget or substitutes approximate sampling.

### Threads, batches and memory

The thread sweep uses eight optimizer updates per window, two windows, identical batch128×2 inputs
and four warmup microbatches. The batch256 comparison uses the same ordered 2,048 spectra per
window and effective batch256; model BatchNorm still means training trajectories can differ.

| Microbatch | Accumulation | Candidate threads | Spectra/s | Peak allocated GPU memory |
| --- | ---: | ---: | ---: | ---: |
| 128 | 2 | 1 | 196.80 | 3.76 GB |
| 128 | 2 | 2 | 229.84 | 3.74 GB |
| 128 | 2 | 4 | 239.54 | 3.76 GB |
| 256 | 1 | 4 | 242.89 | 7.06 GB |
| 512 | 1 | 4 | OOM during warmup | 8 GB card insufficient |

Four threads remain the default; 256 is essentially tied with 128 at this measurement precision.
512 still needs an L40S check. Peak process-tree RSS for the long final test was recorded in JSON;
this sum counts shared parent/worker pages repeatedly, so it is not physical RAM consumption.
An external 80-sample GPU trace during part of the final run has median utilization 46%; it includes
run-boundary samples and has no matched utilization baseline. Throughput remains the decision
metric. Python NVML was unavailable; no package was installed.

Timing compatibility changed: `candidate_topology_assembly_batch_ms` replaces the previous two
metrics and measures one dispatch/join wall interval. Concurrent worker times are never summed.
Packing/output-transfer time now includes serial global-index remapping. The frozen-reference
harness explicitly adapts the two old serial fields into a combined diagnostic; whole-window wall
time is the valid cross-version comparison.

Correctness checks include independent bitmask/heap oracles, extreme float64 selector behavior,
zero bonds, rings, redundant edges, multiword atom sets, strict target overshoot, grouped reuse,
SMARTS and complete candidate/logit-gradient parity at multiple thread counts. The optional GPU
implementation below uses this frozen CPU reference; final validation is recorded in the active
plan as it completes.

## Optional GPU candidate construction (2026-09-17, in progress)

The first `cuda_topology` hybrid retains exact CPU ranking, then builds connectivity,
first-occurrence unique identities and all candidate indices on GPU. The matched 128×2
full-training test reached **254.04 spectra/s**, versus 240.29 for the improved CPU path
(**+5.72%**). Three windows measured 252.93, 254.34 and 254.87 spectra/s; peak allocated GPU
memory was 3.64 GB. It uses the same 16-update/three-window protocol and ordered inputs described
above. The short four-update smoke was 243.62 spectra/s and is not the acceptance measurement.
Artifacts: `output/masshatter_cpu_2026_09_17/msh_b1_long.json` and `msh_b1_smoke.json`.

Integer topology parity passed 42 GPU tests, including direct device CSR, malformed-input
rejection, multiword atom sets, long chains, random graphs and nondefault streams. Nine integration
tests passed, covering complete candidate arrays, live probability gradients and small-model
predictions/loss/parameter gradients. This prototype has two explicit host count/status reads;
physical speculative mask counts are recorded separately from reference logical scan counts.
Lazy native compilation uses the installed toolkit and is reported separately in warmup.

Larger 128-mask tiles and exact pointer jumping in connectivity brought the same hybrid test
to **257.17 spectra/s** (+7.02% over improved CPU; +1.23% over the first hybrid). This is a
small additional gain, with per-window results 257.89, 259.84 and 253.83 spectra/s. The combined
topology/integration suite passed 61 GPU tests after this change. Source snapshot:
`/tmp/msh_b1_optimized_reference`; result: `output/masshatter_cpu_2026_09_17/msh_b1_optimized.json`.
GPU reduction nondeterminism changes subsequent training trajectories slightly; fixed-input
candidate parity is tested independently, and these runs do not establish convergence equivalence.
Native source package data now includes both CUDA kernels; actual cluster compilation remains
unverified. Toolkit discovery honors an explicit argument or `CUDA_HOME`, then a compatible
installed compiler, without modifying the environment.

The exact FP32 single-mixture CUDA selector passed 28 focused tests, including ties,
extreme finite values, multiword masks and direct device-CSR topology parity. The integrated
full-GPU path reached **289.73 spectra/s**: **+20.57%** over improved CPU and **+12.66%** over
the optimized hybrid. Its three windows measured 286.27, 293.38 and 289.62 spectra/s, with
3.65 GB peak allocation. Initial weights and ordered spectrum IDs match the CPU/hybrid tests.
Artifact: `output/masshatter_cpu_2026_09_17/msh_b2_long.json`; frozen candidate sources:
`/tmp/msh_b2_reference`. All 119 combined candidate/model/harness tests passed before timing.

The full path keeps proposal logits and ranked masks on GPU; a regression traps floating-tensor
CPU copies. Static graph metadata remains on the host for validation and bounded allocation.
Ranking has one count read, topology has two count/status reads, and grouped inputs additionally
check equality through a device flag. Host phase timers can include queued encoder work; whole
window throughput is the acceptance metric.

The local B1, B2 and aggregate GPU gates pass. An explicit `mass_shatter_candidate_backend`
setting now defaults to `cpu`, with `cuda` available for the supported FP32/unique/single-mixture
configuration. No trainable tensor names or shapes changed. Three matching opt-in configs and
inference/checkpoint override instructions are in the
[benchmark guide](../../guides/benchmark_masshatter_cpu.md#select-the-optional-cuda-backend).
Public API, real Lightning checkpoint migration and resolved W&B backend logging checks passed.
L40S speed/memory acceptance and long-run convergence remain unmeasured; no remote jobs were
submitted.

### Final public-backend batch matrix

The public `cuda` setting was rerun after integration and per-user cache changes. Every source
hash except the experiment YAML matches across these three runs. Within each batch size,
initial weight hashes, fixed dataset indices and ordered spectrum IDs match the CPU baseline.

| Microbatch | Accumulation | CPU spectra/s | CUDA spectra/s | CUDA gain | CUDA peak allocated |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 2 | 240.29 | **285.88** | **18.97%** | 3.65 GB |
| 256 | 1 | 242.89 | **299.82** | **23.44%** | 6.90 GB |
| 512 | 1 | Warmup OOM | Warmup OOM | Unmeasured | 8 GB card insufficient |

Batch 128 retains the 16-update/three-window protocol; batch 256 uses the matched CPU matrix's
eight-update/two-window protocol. The 512 run failed in GPU view expansion before a measured
window, attempting another 256 MiB with about 183 MiB free. A CPU-only test suite began during
that capacity check, so no 512 timing would have been accepted even if it had fit. The 128/256
throughput windows finished before that suite started.

The public 128 result is also **29.99% faster than the original CPU implementation** in this
full-training harness. This comparison does not multiply in the earlier monitoring-only result,
which used a different diagnostic workload. The public full path remains 11.17% faster than
the optimized hybrid, so both GPU performance gates still pass. The 256 versus 128 difference
is exploratory: windows differ and BatchNorm sees different microbatches.

An external `nvidia-smi --loop-ms=500` trace aligned 86 samples to the three measured 128 windows:
mean whole-device utilization **63.31%**, mean window power 116.6–123.5 W. Device memory includes
other processes. There is no matched CPU utilization trace, so this is an absolute utilization
measurement, not a quantified utilization improvement over CPU or the W&B L40S run.

Artifacts are under `output/masshatter_cpu_2026_09_17/`: `msh_cuda_public_matrix_summary.json`,
`msh_cuda_public_b{128,256,512}.json` and `.log`, plus `msh_cuda_public_b128_gpu.csv`,
`msh_cuda_public_b128_gpu_clock.json` and `msh_cuda_public_b128_gpu_summary.json`. These are local
ignored artifacts. Fixed-input candidate/model differentials passed; ongoing training is not
bitwise deterministic (the second warmup loss differed by about 1.9e-6 before later candidate
trajectories diverged). This does not establish long-run convergence equivalence.

Final validation: the CPU full suite passed **3,422 tests**, with 93 skipped and 2 expected
failures; a separate run with GPU access passed all **118 CUDA-focused tests**. Counts overlap.
Ruff passed on all 25 changed Python files, all three CUDA configs resolved with only the
intended backend/label differences, and documentation links/whitespace checks passed. Test logs
are `output/masshatter_cpu_2026_09_17/msh_final_cpu_suite.log` and `msh_final_cuda_suite.log`.

## Live fast-run follow-up: earlier candidate code still running (2026-09-17)

A fresh read-only W&B query found three fast runs on L40S, all reporting `running`, with
15–16 complete epochs. The original `s3gy1ro4` now reports `crashed` after epoch 15; the cause
was not established. Every fast run has four CPU selection threads and both expensive monitoring
switches disabled. None has `mass_shatter_candidate_backend` in config or the resolved-backend
summary, and their latest epochs still emit separate topology/assembly metrics. Together these
identify the earlier candidate implementation rather than the newly validated CPU/CUDA path.
W&B stores no Git SHA or source snapshot for these runs, so the exact deployed revision is unknown.

| Run | Batch × accumulation | Median epoch interval | Mean sampled GPU utilization | Candidate ms/batch |
| --- | ---: | ---: | ---: | ---: |
| [s3gy1ro4 control](https://wandb.ai/frag-gnn/frag-gnn/runs/s3gy1ro4) | 128×2 | 44.47 min | 23.1% | 417.58 |
| [dny62x0i fast](https://wandb.ai/frag-gnn/frag-gnn/runs/dny62x0i) | 128×2 | 33.52 min | 17.4% | 415.27 |
| [h0ibnndb fast256](https://wandb.ai/frag-gnn/frag-gnn/runs/h0ibnndb) | 256×1 | 32.07 min | 19.0% | 787.48 |
| [du6cr9dx fast512](https://wandb.ai/frag-gnn/frag-gnn/runs/du6cr9dx) | 512×1 | 31.85 min | 19.4% | 1621.31 |

Epoch intervals use common epochs 1–14 and include validation/epoch-end work. Candidate numbers
are medians of training epoch means over epochs 10–14. System history is sampled (3,000 event
rows per run, about 1,500 GPU readings), including training, validation and startup; these are
not phase-isolated GPU averages. Median sampled allocated memory for 128/256/512 is 4.79/8.04/14.76 GB.

The monitoring-only fast128 run shortens epoch wall time by **24.62%**, but its candidate cost
is effectively unchanged. In fast128, selection is 70.79 ms, topology 163.95 ms, assembly
39.67 ms, and packing/transfer 86.57 ms. The old four-thread setting parallelizes selection;
topology/assembly still run serially. Increasing batch size scales candidate work almost linearly,
so 512 reduces epoch time only **0.69%** versus 256 while doubling effective batch size.
Removing GPU-heavy monitoring can lower GPU utilization while improving throughput; utilization
alone must not be interpreted as a slowdown.

Next action: transfer the updated working-tree source to the cluster and create a new job from
it, first retaining CPU to verify the new combined timing metric if desired, then comparing
the opt-in batch256 CUDA config. Changing YAML or syncing source does not update an already
running process: the job generator copies/install sources into its job-local working directory.
New CUDA runs must record `mass_shatter_candidate_backend_resolved: cuda`; new CPU runs must
emit `candidate_topology_assembly_batch_ms` rather than the old separate fields. More selection
threads alone do not address the large serial topology/packing cost in these existing runs.
No runs, configs or jobs were changed by this follow-up. API snapshots and aggregation results
are under `output/masshatter_fast_wandb_2026_09_17/`.

## Batched proposal coverage loss (2026-09-18)

`_proposal_coverage_loss` (`pl_model/mass_shatter_pl.py`) iterated over spectra and built one
dense peak-by-channel match grid per spectrum. It is active in every training step of the
unmerged CMF configs (`mass_shatter_proposal_loss_weight: 0.1`), unlike the monitoring families
already switched off above. It now matches the whole batch at once through
`matched_peak_candidate_pairs` in `model/candidate_support.py`, the same sorted
`(spectrum, m/z rank)` window search the batched channel-support targets use; fragments are
addressed by their global index, so the previous per-spectrum contiguity requirement is gone.
Duplicate channels of one fragment are still counted once per peak, now by deduplicating
`(peak, fragment)` pairs instead of a scatter-add grid.

Local micro-benchmark on this workstation's RTX 2070 SUPER, forward and backward, at the
`dny62x0i` production shapes (batch 128, 50 observed peaks, 233 fragments and about 2,100 valid
ion/H channels per spectrum):

| Implementation | ms per call |
| --- | ---: |
| Per-spectrum loop | 118.6 |
| Batched | 2.4 |

Agreement against the previous loop, kept in `tests/test_mass_shatter_pl.py` as
`_reference_proposal_coverage_loss`: max loss difference 2.4e-6, max gradient difference 7.5e-9
(FP32 reduction order). The randomized equivalence test covers both tolerance modes and both
target modes; separate tests pin cross-spectrum isolation, the no-match case and the empty-input
errors. `tests/test_mass_shatter_pl.py` and `tests/test_mass_shatter_model.py` pass (222 tests).

These are synthetic tensors at production shapes on a smaller GPU than the L40S target. No
end-to-end training-step or epoch measurement was made for this change, so the share of a real
step it removes is not established here; use the existing full-training benchmark for that gate.
