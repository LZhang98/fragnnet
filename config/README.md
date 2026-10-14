# Experiment Configs

## Layout

Experiment configs are grouped by model, with dataset and experiment folders underneath:

```text
config/
  template.yml
  fragnnet/      # FraGNNet (model_type: frag_gnn or fragnnet)
  masster/       # MASSter, including its standalone analysis configs
  neims/
  massformer/
  graff/
  precursor/
```

Debug configs and sweep training configs live under their model too. W&B search definitions
remain in `sweep/`. The shared template and its schema are unchanged. The
model grouping record (removed 2026-10-13; `git show e65339ed:docs/plans/config_model_grouping.md`) explains the path migration;
use the new paths in commands and regenerate previously generated cluster jobs.

A run's configuration resolves in three steps, each overriding the previous:

```
config/template.yml   the defaults for every model
your custom config    the experiment
--override key=value  temporary changes for one invocation
```

**One template, one schema.** `config/template.yml` declares every model's keys, so a FraGNNet
run carries the 73 `masster_*` keys and a MASSter run carries `frag_params` - inert, and
deliberately so. Shared code can then read any key without first asking what model this is, and
a key can never go missing for one model while another model's code still reads it. That failure
mode is what the per-model split (2026-09-20, reverted) kept producing: a key moved into one
model's template while its reader stayed in shared code, breaking every other model at runtime
rather than at validation time.

`model_type` names the model - from the last CLI override, else the custom config, else the checkpoint, else the
template's `frag_gnn`. It selects code, not config; `VALID_MODEL_TYPES` in
[runner.py](../src/fragnnet/runner.py) only guards against a typo in the value. Passing a single
file instead of the directory uses that file directly, which is how a template saved inside a
checkpoint resolves.

### Where a key goes

The template is the shared core first, then one clearly-marked block per model. A model's own
keys stay **flat and prefixed** (`masster_*`, `iceberg_*`, `mf_*`, `graff_*`, `clip_*`,
`mces_*`, `neims_*`) and live in that model's block. Nesting them under a per-model mapping would
invalidate the hparams of every trained checkpoint, which is why the same idea was reverted for
`conditioning_params`. `gnn` and `precursor` have no keys of their own and run entirely on the
shared core.

`tests/test_config_templates.py` enforces this rather than trusting it: a prefixed key must sit
in its own model's block and nowhere else, and the handful of unprefixed keys that belong to a
model (`magma_params`, `magma_dp`, `ann_params`, `ann_fp`, `spectrum_encoder_params`,
`retrieval_eval_size`) are declared there by name. Adding an unprefixed key to a model's block
means editing that list, which is the moment to ask whether it should be prefixed instead.

Precursor BCE and CE-switch BCE are retired. Legacy disabled `precursor_pair_*` and
`ce_pair_*` settings are discarded on load; nonzero loss weights raise an error.

Custom configs are validated against the template, **recursively**: a typo inside `spec_params`
fails as loudly as one at the top level. Mappings whose sub-keys are data rather than schema
(`merge_tolerance_by_inst`, `split_filters`, `inst_type_loss_tol`, `ce_context_ce_key_by_inst`)
are not descended into. Because the template declares every model's keys, a key it does not
declare is a typo - there is no longer an inert-versus-unknown distinction to make, and none
that depends on which files a cluster job happened to stage.

Mapping-valued CLI overrides merge recursively with the existing section and undergo the
same schema checks. For example, `--override 'spec_params={nce: false}'` preserves the
section's other settings; unknown nested keys and scalar replacements of sections fail
during setup. W&B mappings merge recursively too, including new data keys inside the
open mappings, while retaining equal float values used in metric names.

At inference, the checkpoint's migrated hparams are the base. `--custom_fp` overrides
only keys explicitly present in that YAML, with recursive merging of nested sections.
The template validates the overlay; its defaults do not replace saved checkpoint values.
Use the selected run's full training config when explicitly reconstructing those settings.
An overlay cannot select a different model architecture. This policy replaces the earlier
template-over-checkpoint behavior; see the [repair record](../docs/findings/codebase_audit_2026_10_13.md#repairs-2026-10-13).

## Sections

Config is grouped by who consumes it, not by topic:

| Section | Holds |
| --- | --- |
| `spec_params` | what the dataset loads from the spectrum table: merging, cleanup, which metadata columns, the adduct/instrument/mode vocabularies |
| `mol_params` | how a molecule becomes a graph: atom and bond feature lists, fingerprints, preprocessing |
| `frag_params` | FraGNNet's fragment DAG (read only by FraGNNet; inert for every other model) |

Metadata conditioning stays at the top level: `ce_insert_*`, `prec_insert_*`, `inst_insert_*`,
`frag_mode_insert_*`, `context_*`, `int_embedder` and the NCE/ACE normalization statistics.
Grouping them into a `conditioning_params` section was tried and reverted, because it invalidates
every trained checkpoint for no behavioural gain. `spec_params.nce` / `spec_params.ace` are
separate by design: they decide what the *dataset* loads, not how it is embedded.

Note that `mol_params.pyg_node_feats` (atom features: `a_onehot`, `a_degree`, ...) and
`frag_params.pyg_node_feats` (fragment-DAG node features: `cc`, `base_formula`, ...) are
different vocabularies that share a key name. A model only ever reads the block it consumes.


## Setting up a Training Run

To train a model, you can use the following command:

```bash
python scripts/run_pl_model_fit.py -c config/${MODEL}/${DATASET}_${SPLIT}/s${SEED}.yml
```

For example, to train a model on the NIST 2020 dataset using the InChIKey split and the NEIMS model with seed 0, you can use the following command:

```bash
python scripts/run_pl_model_fit.py -c config/neims/nist20_merged_inchikey/s0.yml
```

If you installed the package with Weights and Biases (i.e. using the `dev` option), you can enable logging with a flag:

```bash
python scripts/run_pl_model_fit.py -c config/neims/nist20_merged_inchikey/s0.yml -w online
```

## Available Configs

Configs cover NIST20, MassSpecGym, Spectraverse, NIST23 and other datasets.

### Main Experiment Configs

The following seeds are used:
- SEED=0: 397623
- SEED=1: 634340
- SEED=2: 444303
- SEED=3: 103976
- SEED=4: 132548

FraGNNet-D3: [inchikey configs](./fragnnet/nist20_merged_inchikey/fragnnet_d3/), [scaffold configs](./fragnnet/nist20_merged_scaffold/fragnnet_d3/)

FraGNNet-D4: [inchikey configs](./fragnnet/nist20_merged_inchikey/fragnnet_d4/), [scaffold configs](./fragnnet/nist20_merged_scaffold/fragnnet_d4/)

NEIMS: [inchikey configs](./neims/nist20_merged_inchikey/), [scaffold configs](./neims/nist20_merged_scaffold/)

MassFormer: [inchikey configs](./massformer/nist20_merged_inchikey/), [scaffold configs](./massformer/nist20_merged_scaffold/)

ICEBERG (generator, intensity and optimal-intensity models): retired on 2026-10-11. The local
fork and its configs were removed; run ICEBERG through external
[ms-pred](https://github.com/coleygroup/ms-pred). The configs remain at revision `13b1599d`
(`config/*_merged/iceberg_*`), see the
[reorganization plan](../docs/plans/repo_reorg_plan.md#decided-external-iceberg-dgl-free-main-environment).
The `iceberg_*`, `magma_params` and `magma_dp` template keys stay inert.

### Ablation Experiment Configs

All ablations are performed using the InChIKey split.

The following seeds are used:
- SEED=0: 397623
- SEED=1: 634340
- SEED=2: 444303
- SEED=3: 103976
- SEED=4: 132548

FraGNNet-D3 (+Edges) [configs](./fragnnet/ablations_merged/fraggnn_d3_edges/)
FraGNNet-D3 (-CE) [configs](./fragnnet/ablations_merged/fraggnn_d3_noce/)
FraGNNet-D4 (-CE) [configs](./fragnnet/ablations_merged/fraggnn_d4_noce/)

### Entropy Experiment Configs

All entropy experiments are performed using the Scaffold split, using the FraGNNet-D4 model.

The following seeds are used:
- SEED=0: 397623
- SEED=1: 634340
- SEED=2: 444303
- SEED=3: 103976
- SEED=4: 132548
- SEED=5: 470932
- SEED=6: 499158
- SEED=7: 842766
- SEED=8: 771113
- SEED=9: 742567
- SEED=10: 315874
- SEED=11: 155679
- SEED=12: 885718
- SEED=13: 263526
- SEED=14: 623677


Baseline Ensemble: [configs](./fragnnet/entropy/d4_baseline_ens/)

Low Entropy Ensemble: [configs](./fragnnet/entropy/d4_low_ens/)

High Entropy Ensemble: [configs](./fragnnet/entropy/d4_high_ens/)

## Configuration Parameters

### Fingerprint Parameters (mol_params)

For models using molecular fingerprints (e.g., NEIMS), the following parameters control fingerprint generation:

- **`morgan_radius`** (int, default: 3): Radius for Morgan fingerprint computation. Must be > 0.
  - Original NEIMS paper likely uses radius=2
  - Current optimized default is 3

- **`morgan_nbits`** (int, default: 2048): Number of bits in Morgan fingerprint. Must be > 0.
  - Controls fingerprint vector size
  - Higher values provide more detailed molecular representations but increase memory/computation

- **`rdkit_nbits`** (int, default: 2048): Number of bits in RDKit fingerprint. Must be > 0.
  - Controls RDKit fingerprint vector size
  - Total fingerprint size = morgan_nbits + rdkit_nbits + 167 (MACCS)

**Example:**
```yaml
mol_params:
  fingerprint: True
  fingerprint_morgan: True
  fingerprint_rdkit: True
  fingerprint_maccs: True
  morgan_radius: 3
  morgan_nbits: 2048
  rdkit_nbits: 2048
```

### NEIMS Architecture Parameters

For NEIMS models, the following parameters control the neural network architecture:

- **`neims_bottleneck_factor`** (float, default: 0.5): Bottleneck compression ratio in NeimsBlock layers.
  - Controls the size of intermediate layer: bottleneck_size = bottleneck_factor × hidden_size
  - Lower values (e.g., 0.3) = more compression, fewer parameters, potentially faster training
  - Higher values (e.g., 0.7) = less compression, more capacity, potentially better performance
  - Original NEIMS paper uses 0.5

- **`ff_output_map_size`** (int, default: -1): Low-rank approximation size for output layers.
  - -1 = use standard Linear layers (original NEIMS)
  - >0 = use LowRankDense layers with specified rank (optimized variant)
  - Reduces parameters from ~150M to ~38.7M per output layer when using low-rank

- **`ff_output_activation`** (str, default: "relu"): Output activation function.
  - Options: "relu", "sigmoid"

**Example:**
```yaml
# Original NEIMS configuration
neims_bottleneck_factor: 0.5
ff_output_map_size: -1  # Standard layers
ff_output_activation: relu

# Optimized configuration with low-rank layers
neims_bottleneck_factor: 0.5
ff_output_map_size: 256  # Low-rank approximation
ff_output_activation: relu
```

### FraGNNet Top-K Fragment Competition

FraGNNet can refine its strongest predicted fragments with a transformer before joint
normalization and formula aggregation. Ranking uses each fragment's aggregate valid-channel
logit independently for each spectrum. The transformer receives fragment features, the base
fragment score, and acquisition context, then predicts one scalar residual shared across that
fragment's valid ion/H channels. This changes competition between fragments without disturbing
the base model's relative channel distribution within a fragment.

```yaml
frag_competition: True
frag_competition_top_k: 128
frag_competition_hidden_dim: 64
frag_competition_layers: 2
frag_competition_heads: 4
frag_competition_dropout: 0.1
frag_competition_detach_encoder: True
frag_competition_max_delta: 5.0
```

`frag_competition: False` disables the path. The residual output layer is initialized to zero, so
enabling the transformer on a baseline checkpoint initially preserves its predictions. Fragments
outside the selected top-K retain their base logits and remain in the predicted support. Top-K
selection is non-differentiable. Shared encoder features, base scores, and context are detached by
default to keep the residual branch from destabilizing the base model; the transformer itself
still trains normally. This path currently requires `mlp_output_format: formula`.

The runnable production experiment is
[`fraggnn_d3_ma_mi_nist20_amolc_all_frag_competition.yml`](exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_frag_competition.yml).
Its paired control is
[`fraggnn_d3_ma_mi_nist20_amolc_all_ce_perfrag_feats.yml`](exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_ce_perfrag_feats.yml),
and the head-only K/width/dropout grid is defined in
[`nist20_d3_ma_mi_frag_competition_sweep.yml`](../sweep/nist20_d3_ma_mi_frag_competition_sweep.yml).

### Spectrum Cleanup (`spec_params`)

Spectrum cleanup runs on each source spectrum after split/metadata filtering and before spectra
in the same replicate group are merged. It changes the in-memory training/evaluation targets; it
does not rewrite `spec_df.pkl`.

| Name | Default | Description |
|------|---------|-------------|
| `centroid_peaks` | `False` | Merge nearby signals into an intensity-weighted centroid. |
| `centroid_ppm` | `20.0` | Fixed ppm tolerance used for centroiding. |
| `centroid_use_instrument_tolerance` | `False` | Use the FraGNNet FT/QTOF tolerance and low-mass floor instead of `centroid_ppm`. |
| `centroid_method` | `common_overlap` | Peak grouping rule: `common_overlap` or `naive_centroid`. |
| `filter_subformula_peaks` | `False` | Keep only peaks matching a precursor-compatible molecular subformula. Requires `mol_df.formula`. Candidates always include charge-migration ions, so `[M+Na]+` matches `[frag+H]+` and `[M+Cl]-` matches `[frag-H]-`; this is chemistry only and does not consult the DAG. |
| `subformula_ppm` | `5.0` | Fixed ppm tolerance used for subformula matching. |
| `subformula_use_instrument_tolerance` | `True` | Use the FraGNNet FT/QTOF tolerance and low-mass floor instead of `subformula_ppm`. |
| `drop_empty_spectra` | `True` | Drop spectra, and now-unused molecules, when cleanup removes every peak. |

When both operations are enabled, centroiding runs first and subformula filtering second.

### Target Spectrum Filtering

These options are top-level model hyperparameters handled in `SpectrumPL.preproc_spec()`.
They affect true/target spectra during loss and metric preprocessing; they do not rewrite
the dataset pickle and they do not filter predicted spectra.

| Name | Default | Scope | Description |
|------|---------|-------|-------------|
| `target_rel_ints_thresh` | `0.0` | train true + eval true | Per-spectrum relative threshold in `[0, 1)`. Drops target peaks below `target_rel_ints_thresh * max_int` of that spectrum, e.g. `0.01` keeps only peaks at or above 1% of the base peak. `0.0` drops only zero-intensity peaks. |
| `train_target_top_k_peaks` | `-1` | train true | If positive, keeps only the top-k highest-intensity target peaks per spectrum. `-1` disables it. |
| `train_target_drop_min_int_peak` | `False` | train true | Drops exactly one weakest target peak per spectrum when more than one peak remains. |
| `eval_target_apply_train_rank_filters` | `False` | eval true | When `True`, eval true spectra also apply `train_target_top_k_peaks` and `train_target_drop_min_int_peak`. `target_rel_ints_thresh` already applies to eval true spectra. Falls back to the legacy name `eval_target_apply_train_peak_filters` when absent. |
| `mz_max` | model-specific | train true + eval true | Removes target peaks with `m/z >= mz_max` and also controls bin/loss geometry. |

Filtering order for true spectra is:

```text
target_rel_ints_thresh / mz_max -> train rank filters -> bin/transform/normalize
```

Example:

```yaml
target_rel_ints_thresh: 0.0
train_target_top_k_peaks: 20
train_target_drop_min_int_peak: False
eval_target_apply_train_rank_filters: False
```

With the example above, training loss sees only the top 20 target peaks after the standard
threshold and m/z filters. Evaluation metrics still compare against the normal thresholded
target spectra. Set `eval_target_apply_train_rank_filters: True` only when you want metrics
computed against the same rank-filtered target definition.

### Evaluation Metrics (`auxiliary_scores`)

Controls which similarity/quality metrics are computed during evaluation.
Set in config or passed via `--auxiliary_scores` in `run_save_inference_eval.py`.

#### Binned metrics (require `eval_mz_bin_res`)

Output column name format: `{metric}_{bin_res}` (e.g. `cos_sim_0.01`)

| Name | Description |
|------|-------------|
| `cos_sim` | Cosine similarity between binned predicted and true spectra |
| `jss` | Jensen-Shannon similarity between binned spectra |
| `opt_cos_sim` | Cosine similarity using true peak positions (diagnostic upper bound) |

#### Hungarian matching metrics (global tolerance)

Use rounded m/z values and a fixed matching tolerance.

| Name | Description |
|------|-------------|
| `cos_hun` | Cosine similarity via Hungarian peak matching |
| `jss_hun` | Jensen-Shannon similarity via Hungarian peak matching |

#### Hungarian matching metrics (per-instrument ppm tolerance)

Use per-instrument-type ppm tolerance from `inst_type_loss_tol`. Require `spec_inst_type` in
the batch. Requesting either score while `inst_type_loss_tol` is null raises at setup - it
used to skip the score silently and export -1.0 for every spectrum. They are **opt-in**:
`run_save_inference_eval.py` defaults to `cos_hun`, which needs no table. Reach for the
`_inst_tol` variants when the eval set contains QTOF and the config declares both QTOF in
`spec_params.inst_types` and the table - for an FT-only config they equal the plain versions,
since the FT entry is the global tolerance.

| Name | Description |
|------|-------------|
| `cos_hun_inst_tol` | Cosine via Hungarian matching with per-instrument ppm tolerance |
| `jss_hun_inst_tol` | JSS via Hungarian matching with per-instrument ppm tolerance |

#### Match-based metrics

| Name | Description |
|------|-------------|
| `recall` | Fraction of true peaks matched by a predicted peak |
| `wrecall` | Intensity-weighted recall |
| `precision` | Fraction of predicted peaks matched by a true peak |
| `wprecision` | Intensity-weighted precision |
| `ndcg` | Normalized discounted cumulative gain (union and intersection variants) |

#### Diagnostic / distribution metrics

| Name | Description |
|------|-------------|
| `true_spec_e` | Entropy of the true spectrum |
| `true_spec_ne` | Normalized entropy of the true spectrum |
| `pred_spec_e` | Entropy of the predicted spectrum |
| `pred_spec_ne` | Normalized entropy of the predicted spectrum |
| `true_oos_prob` | True out-of-scope probability |
| `true_oos_e` | True out-of-scope energy |
| `pred_node_count` | Number of predicted fragment nodes |
| `pred_formula_count` | Number of predicted formulas |
| `pred_edge_count` | Number of predicted edges |

#### Sqrt intensity variants

Setting `eval_hun_sqrt: True` enables additional `_sqrt` output columns for all Hungarian
metrics (e.g. `cos_hun_sqrt`, `cos_hun_inst_tol_sqrt`). Intensities are sqrt-transformed
before scoring, which down-weights dominant peaks.

Setting `eval_bin_sqrt: True` does the same for binned metrics (e.g. `cos_sim_sqrt_0.01`).

In `run_save_inference_eval.py` pass `--eval_sqrt True` (controls `eval_hun_sqrt` only).

#### Per-instrument tolerance configuration

Required for `cos_hun_inst_tol` and `jss_hun_inst_tol`, which raise at setup without it.
Feeds the training loss too, not only the scores. An instrument that is not listed falls back
to the global `loss_tolerance_rel` / `loss_tolerance_min_mz`.

```yaml
inst_type_loss_tol:
  FT:   {rel: 1.0e-5, min_mz: 200.0}   # 10 ppm, floor at 200 Da
  QTOF: {rel: 2.0e-5, min_mz: 750.0}   # 20 ppm, floor at 750 Da
```

`rel` is the relative tolerance (ppm as a fraction, so 10 ppm = `1e-5`).
`min_mz` sets a floor m/z below which the absolute tolerance is fixed (avoids
unrealistically tight windows at low mass).

#### Example config snippet

```yaml
auxiliary_scores:
  - cos_hun_inst_tol
  - jss_hun_inst_tol

eval_hun_sqrt: True

inst_type_loss_tol:
  FT:   {rel: 1.0e-5, min_mz: 200.0}
  QTOF: {rel: 2.0e-5, min_mz: 750.0}
```

Produces columns: `cos_hun_inst_tol`, `cos_hun_inst_tol_sqrt`, `jss_hun_inst_tol`, `jss_hun_inst_tol_sqrt`.
