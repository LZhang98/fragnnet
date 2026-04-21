# Recent Changes

---

## 2026-04-21

### `boundary_pair` moved into DAG generation/storage

The `boundary_pair` fragment feature is now precomputed during DAG generation and stored on the
fragment DAG object as:

- `boundary_pair_frag_idxs`
- `boundary_pair_in_local`
- `boundary_pair_out_local`

These tensors are also serialized through HDF5 DAG storage and preserved when the training-time
fragment graph is built.

**Runtime behavior change:**

- The model now expects batched `boundary_pair_*` tensors to come from the loaded DAG data.
- The temporary runtime fallback/cache path was removed.
- If `boundary_pair` is enabled in config, the DAGs must be regenerated with the new code.

**Practical implication:** existing old DAG artifacts without these stored tensors are no longer
supported for `boundary_pair` runs.

**Planned follow-up:** `boundary_pair` itself is intended to be removed in the next version; TODO
notes were added in the model where the feature is instantiated and consumed.

---

## 2026-04-09

### Weighted sampler ablation: inst_weighted / inst_frag_mode_weighted / inst_frag_mode_group_weighted

All three runs hit SLURM time limits (epochs 15, 32, 28 of 100) — results are not converged and
not comparable to the 100-epoch baseline. The weighted samplers produce ~3–4× more mini-batches
per epoch than the baseline (~1,376), making each epoch much slower.

| Variant | mini-batches/epoch | Epochs done | val_cos_sim_0.01 (best) |
|---------|--------------------|-------------|------------------------|
| inst_weighted | ~4,000 | 15 | N/A (too early) |
| inst_frag_mode_weighted | ~4,530 | 32 | 0.6420 |
| inst_frag_mode_group_weighted | ~5,300 | 28 | 0.6083 |

**Why `group_weighted` is much worse even at equal epochs:**

The `adduct_inst_type_frag_mode_group` sampler applies a compound weight:

```
weight = 1 / (class_count × specs_per_group)
```

This doubly-downweights molecules that appear at many CEs. Well-sampled molecules
(e.g. a common HCD/FT/[M+H]+ compound with 5 CE spectra) get weight `~1/(65k × 5)` —
essentially invisible to training. This starves the model of clean signal from the
easy/common regime early in training, leading to noisier gradients and slower convergence.

The plain `adduct_inst_type_frag_mode` sampler only balances by (adduct, inst, frag_mode)
class — a milder intervention that does not destroy the signal from the common regime.

**Conclusion:** Do not use `_group` compound weighting. Need longer wall-time (or
`limit_train_batches`) to properly evaluate any of these weighted samplers.

---

### Edge feature ablation: bcc vs bp vs bp_bcc

Compared three edge feature variants trained on NIST20 (100 epochs, same architecture/seed).

| Variant | Params | Model size | val_cos_sim_0.01 (best) | val_loss_best |
|---------|--------|-----------|------------------------|---------------|
| bcc | 1.7 M | 6.7 MB | 0.6789 | −2.4658 |
| bp | 1.8 M | 7.2 MB | **0.6888** | −2.5097 |
| bp_bcc | 1.9 M | 7.5 MB | 0.6885 | −2.5107 |

**Conclusion: use `bp`.**

- `bp` gains +1.0pp val cos_sim over `bcc`
- `bp_bcc` adds ~100K params over `bp` for negligible val improvement (0.6888 vs 0.6885)
- Bond-path (bp) features are the driver of improvement; bcc features appear redundant when bp is present

---

## 2026-02-24

### Dependency audit from `ccx_run` merge

The `ccx_run` branch introduced two new third-party libraries that were used in code but not
declared in the project's dependency files.

| Library | Version | Where used |
|---|---|---|
| `torch-linear-assignment` | unpinned | `src/fragnnet/utils/spec_utils.py` — batched Hungarian matching in `batch_cos_hun_helper` and `batch_jss_hun_helper` |
| `lmdb` | `1.7.5` | `preproc_scripts/ms2c_inference/01_ms2c_inf_create_candidate_df.py` |
| `pandarallel` | unpinned | `scripts/run_pl_model_fit.py`, `scripts/eval/run_save_inference_eval.py` — parallel pandas apply |

**Files updated:**
- `pyproject.toml` — added `lmdb==1.7.5`, `torch-linear-assignment`, and `pandarallel` to `dependencies`
- `env/env_cc_setup_py310.sh` — added to `pip install pyteomics ...` line
- `env/env_cc_setup_py311.sh` — added to `pip install pyteomics ...` line

### Optional `torch_linear_assignment` with scipy fallback

`torch_linear_assignment` is a GPU-accelerated Hungarian algorithm. It may not be available
on all environments (e.g., CPU-only machines or clusters where the wheel isn't pre-built).

`src/fragnnet/utils/spec_utils.py` was updated so the import is wrapped in a `try/except
ImportError`. A unified helper `_solve_batch_assignment(batch_score)` is defined in both
branches:

- **With `torch_linear_assignment`**: calls `batch_linear_assignment(-batch_score)` +
  `assignment_to_indices` — runs on GPU, fully batched.
- **Fallback (scipy)**: loops over the batch, calls
  `scipy.optimize.linear_sum_assignment(score, maximize=True)` per item, then stacks results
  back into tensors on the original device. Slower and CPU-only, but always available.

A `logging.warning` is emitted at import time when falling back to scipy.
