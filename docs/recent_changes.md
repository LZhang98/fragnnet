# Recent Changes

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
