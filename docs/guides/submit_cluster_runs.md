# Submit training runs on Fir and Vulcan

> **Status:** GUIDE · **Updated:** 2026-10-06
> Submission procedure verified against repository source; no cluster jobs were submitted or tested.

## 1. Prepare on the target cluster

Use [setup_job_cc.py](../../scripts/setup_job_cc.py) from the repository root on the
intended cluster's login node. Selecting a GPU profile changes resource directives; it does
not connect to another cluster. Login hostnames, current account eligibility, and live resource
availability are not established by this repository.

For both Fir and Vulcan, invoke the generator as `python` from the cluster login environment; do
not set or use `FRAGNNET_PYTHON` in cluster commands. The generator uses the standard library;
the generated compute job builds a separate environment.

Have the experiment config and its referenced data, splits, fragment caches, and any pretrained
checkpoint available under the cluster checkout. Use repository-relative config/data paths and
inspect staging commands for the selected model. The generator extracts selected top-level YAML
fields as text; do not assume arbitrary runtime overrides or YAML structures are staged correctly.

The generated job loads `StdEnv/2023`, CUDA 12.2, Python 3.11, and RDKit 2023.09.5 by default,
creates a virtualenv under `${SLURM_TMPDIR}/${SLURM_JOB_ID}`, installs dependencies, stages code
without compiled `.so` files, and builds the package. It needs these local archives:

- `wheels/pyteomics-4.7.5-py3-none-any.whl`
- `wheels/torch_linear_assignment-0.0.6.tar.gz`

Verify modules, wheel availability, and data on the target cluster before submission. See
[INSTALL.md](../../INSTALL.md) for setup and [session workflow](../workflow/session.md) for
experiment provenance. Cluster-generated dependencies differ from the local environment;
record the actual run environment.

## 2. Generate a single-run script

Set these shell variables to the intended config, a unique job name, and an eligible account:

```bash
FRAGNNET_CONFIG=config/path/to/experiment.yml
FRAGNNET_JOB=my_experiment
FRAGNNET_ACCOUNT=aip-rgreiner
```

Replace the example config path. `aip-rgreiner` is used by the existing Vulcan launcher;
choose an account you can use on the target cluster. The generator defaults to `rrg-dwishart`
if the account is omitted. Set the email explicitly to avoid its personal default.

For Fir:

```bash
python scripts/setup_job_cc.py \
  --frag_gnn_dp . \
  --python_script_fp scripts/run_pl_model_fit.py \
  --job_name "$FRAGNNET_JOB" \
  --gpu_profile fir:h100-3g.40gb \
  --cc_account "$FRAGNNET_ACCOUNT" \
  --cc_email your_email@example.org \
  --time 3:00:00 \
  --script_kwargs w=online c="$FRAGNNET_CONFIG" \
  --no_submit=True
```

For Vulcan, use `python` directly:

```bash
python scripts/setup_job_cc.py \
  --frag_gnn_dp . \
  --python_script_fp scripts/run_pl_model_fit.py \
  --job_name "$FRAGNNET_JOB" \
  --gpu_profile vulcan:l40s \
  --cc_account "$FRAGNNET_ACCOUNT" \
  --cc_email your_email@example.org \
  --time 3:00:00 \
  --script_kwargs w=online c="$FRAGNNET_CONFIG" \
  --no_submit=True
```

Do not replace `python` with `FRAGNNET_PYTHON` for cluster submissions.
These are the current **repository presets**, not verified scheduler availability or quota:

| Profile | GPU request (`--gres`) | CPUs | Host memory |
| --- | --- | --- | --- |
| `fir:h100-80gb` | `gpu:h100:1` | 12 | 280G |
| `fir:h100-1g.10gb` | `gpu:h100_1g.10gb:1` | 1 | 35G |
| `fir:h100-2g.20gb` | `gpu:h100_2g.20gb:1` | 3 | 70G |
| `fir:h100-3g.40gb` | `gpu:h100_3g.40gb:1` | 6 | 140G |
| `vulcan:l40s` | `gpu:1` | 6 | 192G |

Profiles override CPU and memory arguments. Check the generated directives and choose resources
and wall time appropriate to the experiment. `w=online` enables W&B logging; arrange authentication
on the cluster. The generator can embed a resolved W&B credential in the generated script, so
keep generated scripts private and exclude credentials from shared records or pasted output.

## 3. Inspect, submit, and record

The generator writes `slurm_scripts/<job_name>.sh` and uses `jobs/` for scheduler logs by default.
Both directories are Git-ignored. Reusing a job name overwrites its generated script.

```bash
bash -n "slurm_scripts/${FRAGNNET_JOB}.sh"
# After reviewing directives, staging paths, environment, and final training command:
sbatch "slurm_scripts/${FRAGNNET_JOB}.sh"
```

Record the returned scheduler job ID, cluster, generated script identity, config, code revision
and dirty state, input identities, and intended outputs in the task record. The job copies code
and inputs when it starts, so edits to the checkout while queued can change what actually runs.
Preserve the intended source/data snapshot through staging and record the executed state.

The generator defaults to `--no_submit=True`. Its `--no_submit` option uses Python `type=bool`,
so the string `False` is also truthy; use the explicit `sbatch` step above to submit.

Inspect the generated `#SBATCH --output` and `--error` paths for logs. W&B output is linked back
to the checkout's `wandb/`; `job_id/` is linked when a job ID is used. Other outputs are not
universally copied back from scratch. In particular, if using `w=disabled`, verify persistent
checkpoint storage before running. Record actual artifact locations and completion/failure
status; scheduler submission alone is not evidence of successful training.

## 4. Cross-validation and existing batch launchers

For Fir, replace the single config argument with the supported CV options:

```bash
python scripts/setup_job_cc.py \
  --frag_gnn_dp . \
  --python_script_fp scripts/run_pl_model_fit.py \
  --job_name fragnnet_cv \
  --cv_config_dp config/fragnnet/spectraverse \
  --cv_start 0 --cv_end 9 \
  --cv_use_job_id \
  --gpu_profile fir:h100-3g.40gb \
  --cc_account "$FRAGNNET_ACCOUNT" \
  --cc_email your_email@example.org \
  --time 24:00:00 \
  --script_kwargs w=online \
  --no_submit=True
```

This generates an array for `cv0.yml` through `cv9.yml`, with per-fold job IDs. Review the
config directory and script, then submit `slurm_scripts/fragnnet_cv.sh` with `sbatch`.
Reusing the same per-fold job IDs may resume prior runs; choose names deliberately.

The existing [Vulcan subformula-filter launcher](../../scripts/manual_slurm/vulcan/submit_fraggnn_d3_subformula_filter.sh)
is a minimal, generation-only example: a single `setup_job_cc.py` invocation that writes the job
script for one config without submitting it. It hard-codes an account and a `vulcan:l40s` profile,
so copy and edit it rather than invoking it as-is. Other launchers submit on invocation: the
[Vulcan architecture launcher](../../scripts/manual_slurm/vulcan/submit_masster_merged_architectures.sh)
validates its config list, generates eight MASSter scripts, then **submits all eight**. It is an
experiment-specific batch launcher, not a generation-only example; inspect its configs, account, and
resource budget before invoking it. Check any launcher for an `sbatch` call and a `--no_submit` flag
before running one. [Fir manual scripts](../../scripts/manual_slurm/fir/) also include preprocessing
jobs; use the applicable procedure rather than treating them as training templates.

## Verification boundary

The examples and resource table were checked against the generator and existing launcher.
Live login access, module resolution, data staging, W&B authentication, queue acceptance, and
training completion remain unverified. This guide documents submission; it does not authorize
an agent to launch experiments outside the user's requested scope.

## Cluster facts from local memory (verified 2026-10-06 where noted)

### Fir GPU profile default

Default every Fir GPU job to `fir:h100-3g.40gb` (`--gres=gpu:h100_3g.40gb:1`, 6 CPUs, 140G; half
an H100). Use `fir:h100-80gb` only with a stated reason, such as a measured OOM at 40 GB. The
presets encode each MIG slice's proportional share, so change gres, CPUs and memory together.
Hand-written `scripts/manual_slurm/fir/*.sh` spell out the triple and use `--gres` with the gres
type, never `--gpus-per-node` with an NVML device name. Exception (user-confirmed 2026-09-21):
the `ma_mi_merged` 256x2 lineage (microbatch 256, `accumulate_grad_batches: 2`,
`loss_batch_size: 32`, `num_workers: 3`; `config/masster/ma_mi_merged/loss_objective_2026_09_21/`,
`scripts/manual_slurm/fir/submit_masster_loss_objective.sh`) is sized for `fir:h100-2g.20gb`
(measured peak 12.7 GiB of about 19.5 GiB usable); do not move it to 3g. Vulcan keeps `vulcan:l40s`.

### Fir environment

- Fir and Vulcan compute nodes have internet (narval does not; use `w=offline` there). For array
  jobs, warm the W&B checkpoint cache once before `sbatch`: `get_best_ckpt_from_wandb` copies
  non-atomically.
- The Fir checkout lives on Lustre `/project` (`/project/6006204/feiw/frag-gnn`; check
  `readlink -f .`). `rrg-dwishart` has no allocation (default 2048 KiB / 1025-inode quota, far
  over), so writes there can fail with `EDQUOT`. `/home` suits a source checkout.
- A Lustre client fault makes `git reset --hard` fail with `unable to create file: File exists`.
  Check `df -hT .` and `git status` first (impossible sizes, or `Cannot send after transport
  endpoint shutdown`). The client is per node, so `ssh login1` usually fixes it. A half-done reset
  leaves `AD` entries, recoverable with `git cat-file blob`. Never `git push --force` over it.
- `setup_job_cc.py` bakes absolute `FRAGNNET_DP` into scripts, so generate scripts on the
  cluster, not locally. One launch hit a Lustre `EIO` on a script file that cleared after `rm`
  and regeneration.

### Vulcan evaluation command (2026-09-16)

```bash
python scripts/setup_job_cc.py --python_script_fp scripts/eval/run_save_inference_eval.py \
  --frag_gnn_dp ./ --time 30:00 --job_name infer_eval_<run_id> --gpu_profile vulcan:l40s \
  --cc_account=aip-rgreiner --script_kwargs "custom_fp=<config>.yml" "wandb_run_id=<id>" \
  "eval_split=test" "use_dynamic_sampler=True" "auxiliary_scores=cos_hun_inst_tol" "eval_sqrt=True"
```

- `--time 30:00` suffices on an L40S for the ~96k-spectrum NIST20 test split; it is a cap, not a
  rate to scale by spectrum count.
- A multi-value flag is one quoted kwarg, e.g. `"auxiliary_scores=cos_sim cos_hun_inst_tol"`.
- Data staging comes from the config (`spec_fp`, `mol_fp`, `split_dp`, `frag_dp`), not the CLI.
- `cos_hun_inst_tol` needs `inst_type_loss_tol` in the run's config.
- Without `--gpu_profile` it falls back to 128G / 8 CPUs.
