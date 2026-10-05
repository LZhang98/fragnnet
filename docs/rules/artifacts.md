# Artifact storage rules

> **Status:** REFERENCE · **Updated:** 2026-10-04
> Where checkpoints and run outputs go, how to name them, and how to cite them. Applies to
> `ckpts/`, `saved_ckpts/`, `output/` and `tmp_profile/`. All four are gitignored and
> unversioned, so a path is the only link between a doc and its evidence.

## 1. What each store is for

| Store | Written by | Holds | Lifetime |
| --- | --- | --- | --- |
| `ckpts/<run_name>/` | `runner.py`, **only when W&B is disabled** (`run_name` = `job_id`, else `wandb_name`) | Local and smoke training checkpoints | Scratch. Can be deleted once the run is read or promoted |
| `wandb/run-*/files/ckpt/` | `runner.py` when W&B is enabled | Checkpoints of every logged run | Owned by W&B. Use the run id, not this path |
| `saved_ckpts/` | `get_best_ckpt_from_wandb` (inference/eval `--model_save_dp` default) and manual pulls | The checkpoints we evaluate and cite | Permanent while any doc cites it |
| `output/<investigation>/` | Analysis, eval and inference scripts | Predictions, tables, plots and logs | Permanent while any doc cites it. Otherwise regenerable |
| `tmp_profile/<run_name>/` | `runner.py` profiler | Profiler traces | Scratch |

**Rule:** a doc may cite `saved_ckpts/`, `output/` or a W&B run id. A doc must never cite
`ckpts/` or `tmp_profile/`. To cite a local run, promote its checkpoint first (§2.3).

## 2. Checkpoints

### 2.1 `ckpts/`: local training

- Give every local run a unique `job_id`: `<arm>_<seed>`, or add a date suffix. Lightning
  does not overwrite a checkpoint with the same name. It writes `model-epoch=000-v1.ckpt`
  beside it, so `-v1` files mean two runs collided. Reusing a name also resumes the old
  checkpoint ([shared-proposal audit](../findings/masster/masster_shared_proposal_pathway_audit_2026_09_26.md)).
- Do not write anything else into `ckpts/`. A loose `ckpts/<run_id>.ckpt` belongs in
  `saved_ckpts/` (§2.2).
- Do not put pytest temp dirs (`--basetemp`) or copies of test checkpoints under
  `ckpts/` or `output/`. Use the scratchpad or pytest's default `tmp_path`.

### 2.2 `saved_ckpts/`: layout

Every entry ends in the W&B run id. Exactly one of three layouts:

1. **Best checkpoint, fetched by the tooling:** `saved_ckpts/<wandb_name>_<run_id>.ckpt`.
   This is the file `get_best_ckpt_from_wandb` writes. Its cache lookup is
   `glob("*_<run_id>.ckpt")` and needs exactly one match. Do not hand-place a second file
   matching that pattern: copies with `_last` or `_ema` suffixes break the cache lookup.
2. **A specific epoch, or a non-best checkpoint:**
   `saved_ckpts/<wandb_name>_<run_id>/ckpt/model-epoch=NNN.ckpt`, plus a `provenance.json`
   next to `ckpt/` that records the run path, remote file, sha256 and config, in the same
   format as `saved_ckpts/k1_ema_j4u25qfd/provenance.json`. Use this layout for `last.ckpt`
   too, rather than a `<name>_last.ckpt` sibling.
3. **A set of runs** (CV folds, a benchmark set): `saved_ckpts/<set_name>/<variant>/`,
   with each file named as in layout 1, for example
   `saved_ckpts/spectraverse/fragnnet/spectraverse_fragnnet_cv0_5pf0qi37.ckpt`.

Never edit a checkpoint in place. These two kinds go under `saved_ckpts/` too, never
`output/`:

- **Derived checkpoints** (weights patched, renamed or converted, such as an instrument
  embedding remap): `saved_ckpts/derived/<source_name>_<run_id>_<change>.ckpt`. Put a
  `<same stem>.json` next to it that gives the source checkpoint's path and sha256, and
  the tracked script under `scripts/` that made it.
- **Third-party checkpoints** (ICEBERG, GLACIER, DreaMS):
  `saved_ckpts/external/<model>/<release>/`. Record the download URL or source commit in a
  `provenance.json`.

### 2.3 Promoting a local checkpoint

When a result from a W&B-disabled run goes into a doc, copy its checkpoint into
`saved_ckpts/` with layout 2. The name is `<job_id>_local_<YYYY_MM_DD>` in place of a run
id. Write `provenance.json` with the git revision, the dirty-diff identity, and the
resolved config. Cite the copy, not `ckpts/`.

## 3. `output/`

- **One directory per investigation:** `output/<topic>_<YYYY_MM_DD>/`. The date is the day
  the investigation starts. Later passes write into the same directory, in subdirectories
  if needed. Do not make a new dated sibling for a rerun of the same question. For a smoke
  run, put `smoke` in the topic (`<topic>_smoke_<date>`).
- **Tool defaults are caches keyed by run id.** `output/inference_eval/<run_id>/` and
  `output/inference_outputs/<run_id>/` are the defaults for `run_save_inference_eval.py`
  and `run_inference.py`. Do not put files from a specific investigation there. Pass
  `--save_dp output/<investigation>/...`. Do not make new undated top-level buckets like
  `analysis/`, `diagnostics/`, `eval/` or `tmp/`.
- **Code goes in the repo, data goes in `output/`.** A script whose result a doc cites goes
  under `scripts/` (usually `scripts/debug/`), where Git tracks it. It does not go inside
  `output/`, because `output/` is unversioned. Shell wrappers and logs may stay with the data.
- **No checkpoints in `output/`.** Use §2.2.
- **No third-party code in `output/`.** Clone external repos (DreaMS, ms-pred) outside the
  artifact stores, and never `pip install -e` from `output/`. An editable install pins
  the folder, and deleting the folder breaks the conda env. (The `dreams` env was
  installed this way from `output/dreams_gate0_2026_10_01/DreaMS/`. Both were removed
  on 2026-10-04.)
- **No scratch in `output/`:** no `tmp/`, `pymp-*`, `__pycache__`, pytest temp folders,
  or copies of the source tree ("stage snapshots"). Use the scratchpad. To keep the code
  state of a run, record its git revision and diff (see the session workflow).
- **Smoke runs are disposable.** No doc should cite a smoke folder. Once the real run has
  landed, the smoke folder is the first thing to propose for deletion.
- **Full prediction dumps are caches, not evidence.** A finding rests on its scored tables
  and summaries. A per-spectrum or per-candidate prediction dump (`predictions_*.hdf5`,
  `*_val.pkl`, `.npz` over 1 GB) can be regenerated from a checkpoint. Put it in
  `cache/`. If a doc really depends on the dump itself, cite the file by name.
- **Name dumps by run, epoch and split** (`i1_i50ig7rf_e059_val.pkl`). Generic names like
  `predictions_test.hdf5` or `spec_df.pkl` match hundreds of lines in a grep, so a
  citation check on them proves nothing.
- **Size.** Before a job writes more than 5 GB, tell the user. Keep large regenerable
  intermediates in a subdirectory named `cache/` so they can be deleted without reading
  the whole tree. To save space, compress. Never downcast values.

## 4. Citing artifacts in docs

- **Checkpoint:** give the W&B run id, the epoch, and the `saved_ckpts/` path, e.g.
  `i1` = `i50ig7rf` epoch 59, `saved_ckpts/i1_inst_ob_obf_i50ig7rf/ckpt/model-epoch=059.ckpt`.
- **Output:** give the investigation directory and the specific file. Name the tracked
  script that regenerates it.
- **Moving a path someone cites:** first
  `grep -rn "<old path>" docs scripts config tests`, then update every hit in the same
  change.

## 5. Deleting and moving

- Agents do not delete, move or overwrite anything in these stores without the user's
  explicit confirmation for that specific path. This includes "obvious" scratch, because
  another session may be training (see `ps`/`nvidia-smi`).
- A path that any doc, script or config cites is never deleted. Moving it follows §4.
- When you propose a cleanup, list each path with its size, how many places cite it
  (`grep -rl "output/<dir>" docs scripts config tests src`), and its last write date.
  Let the user pick. For each top-level folder, cite the folder. For big files inside a
  cited folder, cite each file by name.
- Before deleting a folder, check that no conda env has an editable install pointing at it:
  `grep -rl "<path>" ~/miniconda3/envs/*/lib/python*/site-packages/{__editable__*,*.pth,*.dist-info/direct_url.json}`.
  Also check that no process has files open in it (`lsof +D <path>`).
- Agent shells may block bulk `rm -rf` even after the user approves. In that case, write
  the approved list as a script in the scratchpad and let the user run it. Do not try to
  get around the block.
