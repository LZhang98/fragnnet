# Benchmark MASShatter candidates on L40S

> **Status:** GUIDE · **Updated:** 2026-09-17 · Generation commands checked locally; no cluster
> jobs submitted. Measures training steps; validation/checkpoint overhead needs a later run.

Follow the [cluster prerequisites](submit_cluster_runs.md) for modules, wheels, account and data.
Run these commands with cluster `python`, from the checkout containing the finished CPU changes.
Use a new snapshot directory for each experiment. Keep the snapshots fixed while jobs are queued.

## Freeze source and provenance

The historical CPU source is `a237afe79a24df1347918653e757f43306dcefb8`. Separate snapshots keep
its model/timing interface coherent and let each compute job rebuild its own native extension.
The same new benchmark script is copied into both. Do not copy workstation `.so` files to a cluster.

```bash
set -euo pipefail
FRAGNNET_ORIGIN="$PWD"
FRAGNNET_RUN_ROOT="${SCRATCH:?Set SCRATCH}/msh_cpu_accept_20260917"
FRAGNNET_BASE=a237afe79a24df1347918653e757f43306dcefb8
mkdir -p "$FRAGNNET_RUN_ROOT"/{current,baseline,results,slurm,jobs}
git rev-parse HEAD > "$FRAGNNET_RUN_ROOT/current/source_revision.txt"
git diff HEAD | sha256sum | cut -d' ' -f1 > "$FRAGNNET_RUN_ROOT/current/source_diff_sha256.txt"
rsync -a --exclude='*.so' --exclude='__pycache__' \
  src scripts preproc_scripts config setup.py pyproject.toml "$FRAGNNET_RUN_ROOT/current/"
if [[ -f setup.cfg ]]; then cp setup.cfg "$FRAGNNET_RUN_ROOT/current/"; fi
git archive "$FRAGNNET_BASE" | tar -x -C "$FRAGNNET_RUN_ROOT/baseline"
# Explicit historical candidate sources; no branch switching or checkout changes.
for file in mass_shatter_core.pyx mass_shatter_candidates.py; do
  git show "$FRAGNNET_BASE:src/fragnnet/frag/$file" \
    > "$FRAGNNET_RUN_ROOT/baseline/src/fragnnet/frag/$file"
done
cp scripts/debug/benchmark_mass_shatter_full_training.py \
  "$FRAGNNET_RUN_ROOT/baseline/scripts/debug/"
printf '%s\n' "$FRAGNNET_BASE" > "$FRAGNNET_RUN_ROOT/baseline/source_revision.txt"
printf '' | sha256sum | cut -d' ' -f1 > "$FRAGNNET_RUN_ROOT/baseline/source_diff_sha256.txt"
for variant in baseline current; do
  ln -s "$FRAGNNET_ORIGIN/data" "$FRAGNNET_RUN_ROOT/$variant/data"
  ln -s "$FRAGNNET_ORIGIN/wheels" "$FRAGNNET_RUN_ROOT/$variant/wheels"
done
```

The archived tracked baseline has an empty tracked diff; the copied benchmark is separately
hashed in its JSON. Historical SHA256 checks are `92061de16c98aba6ee8be6163cfab57eebb5fb32f9d910a5d99ee63cdc7667e8`
for the LF `.pyx` blob and `fcb26b2bc6c2bea9ff3adb9802419304d7a984eeb9e35a86de9e4819eb2a61cc`
for candidate Python. The local reference `.pyx` has CRLF endings; its normalized text matches.
Data and wheels remain shared: preserve their contents through the comparison.

## Generate three three-hour jobs per source variant

Set your eligible account/email and choose `baseline` or `current`. Repeat for the other variant.
`c=...` is required for the generator to discover the MASShatter data/split paths; the benchmark
accepts the resulting `-c`. Hyphenated benchmark options pass through `script_kwargs` unchanged.

```bash
FRAGNNET_VARIANT=current
FRAGNNET_ACCOUNT=aip-rgreiner
FRAGNNET_EMAIL=your_email@example.org
cd "$FRAGNNET_RUN_ROOT/$FRAGNNET_VARIANT"
FRAGNNET_REVISION=$(cat source_revision.txt)
FRAGNNET_DIFF=$(cat source_diff_sha256.txt)
for batch in 128 256 512; do
  config=config/massshatter/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_cmf_dual_ce_fast
  accumulation=1
  if [[ "$batch" == 128 ]]; then accumulation=2; else config="${config}_batch${batch}"; fi
  config="${config}.yml"
  job="msh_cpu_${FRAGNNET_VARIANT}_b${batch}"
  python scripts/setup_job_cc.py \
    --frag_gnn_dp "$PWD" --python_script_fp scripts/debug/benchmark_mass_shatter_full_training.py \
    --job_name "$job" --script_dp "$FRAGNNET_RUN_ROOT/slurm" --job_dp "$FRAGNNET_RUN_ROOT/jobs" \
    --gpu_profile vulcan:l40s --cc_account "$FRAGNNET_ACCOUNT" --cc_email "$FRAGNNET_EMAIL" \
    --time 3:00:00 --no_submit=True \
    --script_kwargs c="$config" label="$job" output="$FRAGNNET_RUN_ROOT/results/${job}.json" \
    batch-size="$batch" accumulation="$accumulation" candidate-threads=4 num-workers=6 num-threads=2 \
    subsample-size=8192 optimizer-steps=16 warmup-optimizer-steps=16 repeats=5 device=cuda \
    source-revision="$FRAGNNET_REVISION" source-diff-sha256="$FRAGNNET_DIFF"
  bash -n "$FRAGNNET_RUN_ROOT/slurm/${job}.sh"
done
```

Inspect directives, paths, environment and final commands, then submit explicitly if intended:

```bash
for batch in 128 256 512; do
  sbatch "$FRAGNNET_RUN_ROOT/slurm/msh_cpu_${FRAGNNET_VARIANT}_b${batch}.sh"
done
```

The generator stages source without `.so` files and builds it inside the compute environment.
This separate-snapshot method needs no `--reference-dir`. That option instead requires a native
reference built with the **target** Python/NumPy/compiler environment plus its matching candidate
Python file; copying the local reference binary is not a substitute. JSON outputs use persistent
absolute paths because this generator does not copy benchmark output back from scratch.

Compare within batch size: 128×2 and 256×1 both use effective batch 256; 512×1 is separate.
Verify identical ordered spectrum IDs, initial weight hashes and resolved optimizer settings;
compare loss/count trajectories, throughput and memory. Repeat with reversed source order on
matched L40S hardware. Window throughput includes loading, transfer, backward and optimizer;
latencies default to host enqueue timings. NVML utilization can be unavailable; collect
`nvidia-smi` externally when needed. Input bytes are logical tensor sizes, not measured PCIe traffic.

Sweep `candidate-threads=1/2/4` on the current snapshot with new labels/output paths to measure
thread scaling under the same six-worker/six-CPU allocation. Threads=1 still uses current code,
so it is not the historical baseline. Apply the [CPU plan's gates](../plans/masshatter_candidate_backend_plan.md#stage-0-establish-the-acceptance-benchmark)
and record regressions/OOMs as well as improvements; higher GPU utilization alone is insufficient.

## Select the optional CUDA backend

The training/validation setting is `mass_shatter_candidate_backend: cpu` (default) or `cuda`.
CUDA is opt-in: local correctness and full-training speed gates passed, while target L40S
acceptance remains pending. It performs exact mask selection and topology on GPU, preserving
the candidate and checkpoint tensor contracts. Immutable host graph metadata and small ragged
count/status reads remain; this is not a synchronization-free or CUDA-graph-captured path.

Matched CUDA configs sit beside the existing CPU configs:

- `nist20_ma_mi_unmerged_cmf_dual_ce_fast_cuda.yml`: 128×2.
- `nist20_ma_mi_unmerged_cmf_dual_ce_fast_batch256_cuda.yml`: 256×1.
- `nist20_ma_mi_unmerged_cmf_dual_ce_fast_batch512_cuda.yml`: 512×1, effective batch doubles.

All are under `config/massshatter/ma_mi_unmerged/cmf/`; optimizer/loss/candidate settings match
their CPU counterparts. Older checkpoints missing the backend migrate explicitly to CPU;
saved values and explicit checkpoint-loader overrides are preserved. For inference, pass
`--mass_shatter_candidate_backend cpu` or `cuda` to `scripts/run_inference.py` to change only
this setting after extracting checkpoint configuration. A CUDA checkpoint can therefore be
used on CPU by explicitly selecting `cpu`. Training records the resolved setting after W&B
resume configuration merging; check that value when resuming an older run.

## Compare CPU, hybrid and full CUDA construction

Reuse the frozen `current` snapshot above, provided it includes the CUDA Python modules, native
`.cu` sources and their `pyproject.toml` package-data entry. If it predates those files, create a
new snapshot and provenance record; leave snapshots used by queued jobs fixed. These are the
benchmark harness's `--candidate-backend` choices, independent of the model's device:

| Choice | Candidate computation |
| --- | --- |
| `cpu` | Improved CPU selection, topology and numeric assembly; reference for GPU comparisons. |
| `cuda_topology` | Hybrid: CPU selection, then GPU topology, deduplication and numeric assembly. |
| `cuda_full` | Public `cuda` backend: GPU selection, topology, deduplication and numeric assembly. |

Both CUDA variants require FP32, one proposal mixture, unique-fragment mode, zero probability
floors and SMARTS disabled; the fast configs satisfy these restrictions. Unsupported inputs and
workspace limits raise errors. They do not silently switch to CPU. The harness resolves CPU
for `cpu`/`cuda_topology` and CUDA for `cuda_full`, overriding the YAML backend to match the
benchmark label. Keep target-hardware promotion gated against the improved CPU under the
[backend plan](../plans/masshatter_candidate_backend_plan.md#phase-b-develop-the-optional-gpu-solution).

Current native bounds are 512 atoms and 2,048 bonds per molecule, 65,536 masks after forced
merging, and a 512 MiB scratch limit per native stage. The full selector additionally limits
the learned pool to 65,536 masks; adding forced masks can exceed the topology bound and raises
an explicit error. Final candidate tensors and model activations are outside those scratch limits.

On the compute node, optional native compilation needs an existing `nvcc`, a compatible host
C++ compiler, CUDA headers/libraries, and writable node-local `/tmp`. The generated job loads
CUDA 12.2; confirm that the installed PyTorch CUDA major version matches the loaded toolkit.
Discovery honors `CUDA_HOME` before `nvcc` on `PATH`; an explicit incompatible `CUDA_HOME`
raises an error. Set it to the loaded toolkit root when needed, after module loading in the
generated script. Compilation installs no packages and requires no Ninja. Native binaries are
cached under `/tmp/fragnnet_cuda_topology_<user-hash>` and
`/tmp/fragnnet_cuda_selection_<user-hash>`, where the hash identifies the user's home directory.
These per-user paths avoid ownership conflicts on shared nodes. Keep caches local to the
compute node and do not copy workstation binaries into the snapshot. The first
candidate call normally compiles during warmup; JSON native-load diagnostics include that cost,
while measured windows follow warmup. GPU candidates still synchronize for ragged output counts.

Generate nine scripts from the same current snapshot, with identical batches and worker counts:

```bash
cd "$FRAGNNET_RUN_ROOT/current"
FRAGNNET_REVISION=$(cat source_revision.txt)
FRAGNNET_DIFF=$(cat source_diff_sha256.txt)
for backend in cpu cuda_topology cuda_full; do
  for batch in 128 256 512; do
    config=config/massshatter/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_cmf_dual_ce_fast
    accumulation=1
    if [[ "$batch" == 128 ]]; then accumulation=2; else config="${config}_batch${batch}"; fi
    config="${config}.yml"
    job="msh_backend_${backend}_b${batch}"
    python scripts/setup_job_cc.py \
      --frag_gnn_dp "$PWD" --python_script_fp scripts/debug/benchmark_mass_shatter_full_training.py \
      --job_name "$job" --script_dp "$FRAGNNET_RUN_ROOT/slurm" --job_dp "$FRAGNNET_RUN_ROOT/jobs" \
      --gpu_profile vulcan:l40s --cc_account "$FRAGNNET_ACCOUNT" --cc_email "$FRAGNNET_EMAIL" \
      --time 3:00:00 --no_submit=True \
      --script_kwargs c="$config" label="$job" output="$FRAGNNET_RUN_ROOT/results/${job}.json" \
      candidate-backend="$backend" batch-size="$batch" accumulation="$accumulation" \
      candidate-threads=4 num-workers=6 num-threads=2 subsample-size=8192 \
      optimizer-steps=16 warmup-optimizer-steps=16 repeats=5 device=cuda \
      source-revision="$FRAGNNET_REVISION" source-diff-sha256="$FRAGNNET_DIFF"
    bash -n "$FRAGNNET_RUN_ROOT/slurm/${job}.sh"
  done
done
```

This block generates scripts only. Review them before any separate submission. Compare each
CUDA variant with `cpu` at the same batch size, verify candidate/count/loss trajectories and
source hashes, and reverse backend order for the repeat comparison. Report OOMs and cold build
costs alongside throughput and memory. These windows omit validation/checkpoint overhead;
L40S results and a subsequent matched training run remain the promotion checks.
