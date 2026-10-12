# FraGNNetPL Training Loop Memory Review

> **Status:** REFERENCE · **Updated:** 2026-10-06
>
> Memory pressure points in the Lightning training loop.

This note documents the main memory pressure points found while reviewing the
`FraGNNetPL` training loop and the shared `SpectrumPL` step logic.

The findings fall into two categories:

- GPU memory used during a single forward/backward step
- CPU RAM used by dataloading and epoch-level result retention

## 1. Eval preprocessing is built on the autograd path

Relevant code:

- `src/fragnnet/pl_model/spectrum_pl.py:1145`
- `src/fragnnet/pl_model/spectrum_pl.py:1150`
- `src/fragnnet/pl_model/spectrum_pl.py:1219`
- `src/fragnnet/pl_model/spectrum_pl.py:1267`

### What happens

`_common_step()` first prepares tensors needed for the training loss:

- `pred_mzs`
- `pred_logprobs`
- `train_pred_mzs`
- `train_pred_logprobs`

Those tensors must stay attached to autograd because they contribute to
`loss_d["loss"]` and are used for `backward()`.

Later in the same method, `_common_step()` builds a second set of tensors for
evaluation and metric computation:

- `eval_true_mzs`
- `eval_true_probs`
- `eval_pred_mzs`
- `eval_pred_probs`

The metric function itself is called under `with th.inference_mode():`, but the
eval tensors are created before that block. Since `eval_pred_*` is derived from
model outputs, PyTorch can keep extra graph-connected intermediates alive until
the step finishes even though those tensors are only needed for metrics.

### Why it costs memory

This effectively creates two views of the prediction path in the same step:

1. a training path used for the loss
2. an eval path used only for metrics/logging

Only the first path needs gradients. If the second path is created with grad
tracking still enabled, GPU memory can stay higher than necessary because the
autograd engine cannot immediately discard all intermediates.

### How to improve it

Best options:

- Move the entire eval preprocessing block under `torch.no_grad()` or
  `torch.inference_mode()`
- Or explicitly call `.detach()` on `pred_mzs` and `pred_logprobs` before the
  eval-preprocessing branch

The key idea is simple: if a tensor is used only for metrics or logging, build
it outside autograd.

## 2. Full per-batch outputs are retained for the entire epoch

Relevant code:

- `src/fragnnet/pl_model/spectrum_pl.py:1454`
- `src/fragnnet/pl_model/spectrum_pl.py:1493`
- `src/fragnnet/pl_model/spectrum_pl.py:1532`
- `src/fragnnet/pl_model/spectrum_pl.py:1548`

### What happens

`_update_results()` stores batch outputs on CPU for later epoch-end reduction.
That includes:

- `true_mzs`
- `true_logprobs`
- `pred_mzs`
- `pred_logprobs`
- per-sample ids
- strings such as SMILES and instrument/adduct labels
- every tracked metric tensor

Each batch is appended into Python lists inside `train_results`, `val_results`,
or `test_results`.

At epoch end, `_consolidate_results()` concatenates all those per-batch tensors
into one large tensor per key.

### Why it costs memory

There are two layers of memory pressure:

1. During the epoch, RAM grows steadily because every batch is retained.
2. At epoch end, concatenation allocates new large tensors while the old batch
   chunks are still alive, so peak memory becomes "all batch chunks" plus "the
   final concatenated tensor".

This is especially expensive here because the retained objects include ragged
spectra for both ground truth and predictions, not just scalar losses.

### How to improve it

For metrics like mean and standard deviation, do not retain every example.
Instead:

- keep running sums / weighted sums / weighted squared sums
- update them batch by batch
- only retain full spectra if they are required for image logging or detailed
  test-time export

The biggest win is to avoid storing raw train-epoch spectra by default.

## 3. Train batches always run the eval metric path

Relevant code:

- `src/fragnnet/pl_model/spectrum_pl.py:1217`
- `src/fragnnet/pl_model/spectrum_pl.py:1267`
- `src/fragnnet/pl_model/spectrum_pl.py:1327`

### What happens

Even for `split="train"`, `_common_step()`:

- preprocesses eval spectra
- prepares metric inputs
- runs `metric_fn`
- stores metric outputs for later epoch reduction

This means every training batch pays the cost of both optimization work and
evaluation/logging work.

### Why it costs memory

The extra path creates additional tensors and increases temporary working-set
size on both GPU and CPU. It also feeds more data into the epoch-level result
accumulator described above.

### How to improve it

Good options:

- Skip most auxiliary metrics for `split == "train"`
- Compute train metrics only every `N` steps
- Keep train-time logging limited to loss and a small number of scalar metrics

If the purpose of training metrics is only trend tracking, sparse logging is
usually enough.

## 4. Pairwise losses materialize full batch-by-batch matrices

Relevant code:

- `src/fragnnet/models/fragnnet/fragnnet_pl.py:920`
- `src/fragnnet/models/fragnnet/fragnnet_pl.py:947`
- `src/fragnnet/models/fragnnet/fragnnet_pl.py:976`
- `src/fragnnet/model/loss.py:1200`
- `src/fragnnet/model/loss.py:1300`
- `src/fragnnet/model/loss.py:1447`

### What happens

The pairwise regularizers compute spectrum similarity matrices for the whole
batch:

- pairwise cosine similarity
- pairwise JSS similarity
- pairwise cross-entropy

The helper functions chunk the dense spectrum representation, but they still
allocate a final output matrix of shape `[batch_size, batch_size]`.

### Why it costs memory

The total output size scales quadratically with batch size. If dynamic batching
allows larger packed batches, these matrices can become one of the dominant
temporary allocations in the step.

Some helpers also densify chunk blocks, which increases peak memory further.

### How to improve it

Possible improvements:

- Disable pairwise losses for large-batch runs
- Use a smaller `loss_batch_size` chunk
- Rewrite the pairwise objective to accumulate the final MSE blockwise instead
  of storing the full similarity matrix

If only the scalar MSE is needed, there is no strict need to keep the full
matrix in memory.

## 5. `track_datapoint_metrics` preallocates large CPU buffers

Relevant code:

- `src/fragnnet/pl_model/spectrum_pl.py:980`
- `src/fragnnet/pl_model/spectrum_pl.py:987`
- `src/fragnnet/pl_model/spectrum_pl.py:995`

### What happens

When `track_datapoint_metrics` is enabled, `_setup_result_trackers()` allocates
large fixed-size CPU tensors for every metric and every split.

The reserved size is based on `max_num_datapoints`, which can be large:

- around `22000` when merged
- around `270000` otherwise

### Why it costs memory

This memory is reserved up front, before a single epoch finishes. If many
metrics are tracked, the total reserved RAM can become substantial.

### How to improve it

Safer options:

- Keep `track_datapoint_metrics=False` unless it is actively needed
- Allocate these buffers lazily only when a best-epoch export is required
- Restrict datapoint tracking to validation or test only

This is a CPU-side cost, not a GPU-side cost.

## 6. Dataloader worker prefetch can amplify RAM usage

Relevant code:

- `src/fragnnet/runner.py:161`
- `src/fragnnet/runner.py:266`
- `src/fragnnet/runner.py:634`

### What happens

`init_dataloader()` sets:

- `num_workers`
- `pin_memory`

but does not explicitly set lower-memory controls such as `prefetch_factor`.

For heavy fragment batches, each worker may hold multiple prefetched batches at
once. With several workers, host RAM can rise quickly even before the model sees
the data.

The trainer already forces dataloader reloads each epoch in some modes to limit
worker memory growth, which suggests dataloader RAM has been a known issue.

### Why it costs memory

The memory footprint is roughly:

- dataset object state per worker process
- collated batch objects waiting in worker queues
- pinned-memory copies ready for GPU transfer

Large ragged graph/spectrum batches make this more expensive than in a standard
image pipeline.

### How to improve it

Try:

- reducing `num_workers`
- setting `prefetch_factor=1`
- disabling `pin_memory` when GPU transfer is not the bottleneck

This is mainly a host-RAM optimization.

## Priority order

If memory problems need to be fixed in order of likely impact:

1. Detach or `no_grad` the eval-preprocessing path
2. Stop retaining full epoch outputs unless they are explicitly needed
3. Reduce or gate train-time metric computation
4. Rework or disable pairwise losses for large batches
5. Disable `track_datapoint_metrics` unless required
6. Tune dataloader workers and prefetching

## Practical split between GPU and CPU issues

Mostly GPU-step memory:

- eval preprocessing on the autograd path
- pairwise similarity losses
- extra train-time metric preprocessing

Mostly CPU RAM:

- epoch-long result retention
- datapoint metric preallocation
- dataloader workers and prefetch queues

## Training memory levers (from local memory, 2026-07-14; flags re-checked 2026-10-06)

- FraGNNet training memory is activation-bound, not parameter-bound (about 5.8M parameters).
  Activations scale with `mlp_hidden_size` x total DAG nodes in the batch x `mlp_output_dim`, so
  the dials are `dynamic_batch_sampler_max` (the master dial; it overrides `train_batch_size`) and
  `mlp_hidden_size`.
- `activation_checkpointing: True` (default False) saves about half the activation memory for
  one extra forward.
- `skip_train_explainability: True` skips the `frag_real_*` / `nb_iso` marginals on training
  forwards only; spectrum outputs are bit-identical; it raises if an entropy loss is active.
  Tests: `tests/test_skip_train_explainability.py`.
- Not implemented: checkpointing only `formula_module`, which has a better memory/compute ratio.
- bf16 is a no-go (unstable; user decision 2026-07-14).
