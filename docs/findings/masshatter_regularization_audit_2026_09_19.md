# MASShatter regularization audit: dropout, normalization, generalization

> **Status:** FINDING · **Updated:** 2026-09-19 (America/Chicago)
> Regularization and optimization have effectively never been varied in MASShatter, and the
> train/eval candidate-set divergence is driven by BatchNorm rather than by dropout.
> The atom decoder was removed as a result (3.5); no other model code changed.

## Scope and method

Source checkout `eea9d623`, clean at audit start, branch `research/masshatter-ion-h-selection`.
Three evidence sources:

1. All 156 config files under `config/massshatter/**`.
2. A read-only W&B query of `frag-gnn/frag-gnn` filtered to `config.model_type = mass_shatter`,
   run 2026-09-19: **226 runs**, 192 merged and 34 unmerged. This supersedes the 197-run
   population in the [unmerged overfitting audit](masshatter_unmerged_overfitting_2026_09_16.md).
3. Ten mechanism probes, reproducible with
   `scripts/debug/probe_masshatter_regularization.py` (add `--wandb` for the run census).

**What the probes do and do not establish.** Each probe builds a randomly initialized,
production-shaped module (GINE_GSA 384x2, head 512, bond head 3 layers, intensity transformer
384x3 / ff1024 / 128 inducing points, atom decoder 2 except where stated) and measures one
mechanism. They establish that a mechanism exists and its sign. They are **not** calibrated magnitudes for a
trained model: weights are random, the probe batch is two molecules, and BatchNorm warm-up
uses synthetic features. No training run or checkpoint evaluation was performed. The only
model-code change is the atom-decoder removal recorded in 3.5.

## Headline

Of 226 MASShatter runs, exactly **two** vary a regularization or optimization setting without
simultaneously changing the architecture, and both predate the merged-target change, so
neither is usable as evidence today. Separately, the divergence between the candidate set
MASShatter optimizes in training and the one it is evaluated on is driven by **BatchNorm, not
dropout**: at dropout 0 the agreement is Jaccard 0.317/0.611 under `mol_normalization: batch`
and exactly 1.000 under `layer`, `graph` and `none`. That configuration is used in 213 of 226
runs.

## 1. Dropout

### 1.1 The corpus is one point in regularization space

Config files (156) and W&B runs (226) agree. Counts below are runs.

| Setting | Values observed | Verdict |
|---|---|---|
| `mass_shatter_head_dropout` | 0.1 x226 | never varied |
| `weight_decay` | 0.01 x226 | never varied |
| `lr` | 0.001 x226 | never varied |
| `gradient_clip_val` | 5 x226 | never varied |
| `train_sample_weight` | none x226 | never varied |
| `lr_schedule_type` | exponential x226 | never varied |
| `mass_shatter_intensity_transformer_dropout` | 0.1 x207, absent x19 | never varied |
| `mass_shatter_context_dropout` | absent x220, 0 x6 | **never nonzero** |
| `mass_shatter_context_normalization` | absent x220, none x6 | **never `layer`** |
| `lr_schedule` | False x225, True x1 | one run |
| `mol_normalization` | batch x213, layer x13 | see 1.2 |
| `mol_dropout` | 0.1 x219, 0.05 x7 | see 1.2 |

### 1.2 The apparent variation is confounded by architecture

Cross-tabulating the two settings that do vary against `mol_gnn_type`:

- All **7** `mol_dropout: 0.05` runs are Graphormer. Dropout co-varies with the encoder.
- **12 of 13** `mol_normalization: layer` runs are Graphormer or GINE_GRAPHORMER, where
  `layer` is *forced*: `Graphormer.__init__` raises `"Graphormer requires mol_normalization:
  layer"` ([nn_blocks.py:403](../../src/fragnnet/model/nn_blocks.py#L403)).

That leaves two unconfounded runs, both created 2026-08-18 with `spec_params.merge_method`
absent, i.e. before `tolerance_consensus` changed the merged validation target around
2026-09-03. Cosines are not comparable across that boundary, and both sit at ~0.70 while the
current merged leaders sit at ~0.80.

| Run | Variation | Batch size | Best cosine | Nearest sibling |
|---|---|---|---|---|
| [qz0uelf0](https://wandb.ai/frag-gnn/frag-gnn/runs/qz0uelf0) | `lr_schedule: True`, warmup 5000, `lr_decay_rate: 1` (warmup only, no decay) | 128 | 0.7014 | [h7wsvtcu](https://wandb.ai/frag-gnn/frag-gnn/runs/h7wsvtcu) 0.6894 at batch 8 |
| [2bnhwtof](https://wandb.ai/frag-gnn/frag-gnn/runs/2bnhwtof) | `mol_normalization: layer` | 64 | 0.6962 | [87g7ybuj](https://wandb.ai/frag-gnn/frag-gnn/runs/87g7ybuj) 0.6987 at batch 8, [yt48svg5](https://wandb.ai/frag-gnn/frag-gnn/runs/yt48svg5) 0.7061 at batch 64 but with bond curriculum |

Neither has a batch-size-matched control, so neither supports a conclusion. `qz0uelf0` is the
best run in its family, but its batch size differs from its baseline by 16x. The correct
reading is that **no clean regularization or schedule comparison exists**, not that layer
normalization is neutral or that warmup helps.

### 1.3 Five prepared ablation configs have never been launched

These exist on disk and are absent from all 226 runs:

| Config | Setting | Runs |
|---|---|---|
| `ma_mi_merged/..._lb64_drop02.yml` | all three dropouts 0.2 | 0 |
| `ma_mi_merged/..._lb64_drop03.yml` | all three dropouts 0.3 | 0 |
| `ma_mi_merged/..._lb64_wd10.yml` | `weight_decay: 0.1` | 0 |
| `ma_mi_merged/..._lb64_graphnorm.yml` | `mol_normalization: graph` | 0 |
| `ma_mi_merged/..._lb64_ctxnorm.yml` | `mass_shatter_context_normalization: layer` | 0 |

### 1.4 The dropout mechanics themselves are correct

- **Placement (probe 2).** `F.relu(dropout(x))` equals `dropout(F.relu(x))` exactly
  (max difference 0.000e+00), because ReLU is positively homogeneous. The GINE residual at
  [nn_blocks.py:846](../../src/fragnnet/model/nn_blocks.py#L846) is not a bug.
- **Coverage (probe 1).** The production encoder holds 2 `nn.Dropout` modules, both at p=0.1.
  There is no dropout on `input_project`, so MASShatter has no input-feature dropout analogue
  of FraGNNet's `cc_atom_dropout`.
- **Test coverage.** Every model test builds with `precursor_normalization="layer"` and
  `dropout=0.0` ([test_mass_shatter_model.py:93-109](../../tests/test_mass_shatter_model.py#L93-L109)).
  The production combination (BatchNorm, dropout 0.1) is untested.

## 2. Normalization

### 2.1 BatchNorm leaks batch composition into a molecule's embedding (probe 4)

Molecule 0 is embedded twice with a byte-identical subgraph: once among 19 companions, once
alone. Relative change in its atom embeddings:

| `mol_normalization` | companions move molecule 0 | train vs eval, same batch |
|---|---:|---:|
| batch | **1.0868** | **0.1136** |
| layer | 0.0000 | 0.0000 |
| graph | 0.0000 | 0.0000 |
| none | 0.0000 | 0.0000 |

A molecule's representation moves by more than its own norm depending on which molecules share
its batch. Holding molecule 0 fixed and growing its companions from 30 to 120 atoms moves it by
0.7900 relative; the other three are exactly 0. The train/eval column uses running statistics
converged over 300 batches, so 0.1136 is a steady-state gap, not a warm-up transient.

### 2.2 That leakage reaches candidate selection (probe 5)

Jaccard agreement between the m/z set MASShatter selects in training mode and in evaluation
mode, with BatchNorm statistics converged on 150 varied 64-molecule batches. mol0 is aspirin
(381 evaluation candidates), mol1 is caffeine (582).

| `mol_normalization` | dropout | mol0 | mol1 |
|---|---:|---:|---:|
| batch | 0.0 | **0.317** | **0.611** |
| batch | 0.1 | 0.601 | 0.771 |
| layer | 0.0 | 1.000 | 1.000 |
| layer | 0.1 | 0.983 | 0.838 |
| graph | 0.0 | 1.000 | 1.000 |
| graph | 0.1 | 0.966 | 0.743 |
| none | 0.0 | 1.000 | 1.000 |
| none | 0.1 | 0.994 | 0.890 |

Two separate mechanisms, cleanly isolated:

- **BatchNorm produces a divergence with no stochasticity at all.** At dropout 0 the three
  other normalizations give exact agreement while `batch` gives 0.317/0.611. Training
  optimizes over a candidate set that evaluation systematically never sees, and the model is
  selected on the evaluation-mode metric.
- **Dropout adds a smaller, stochastic perturbation** (norm=none, p=0.1: 0.994/0.890). It is
  real but secondary, and it is the kind that averages out across steps.

Interestingly, adding dropout *raises* agreement under BatchNorm (0.317 to 0.601). Nothing
should be read into the direction; it indicates the two interact rather than compose.

These divergences are probably over-estimates: a two-molecule probe batch maximizes the gap
between that batch's atom statistics and the population statistics held in the running buffers,
and the weights are random rather than adapted to train-mode statistics. They are not
guaranteed over-estimates, though — warming the running statistics on a varied distribution
made the `batch` divergence *worse*, not better (mol0 Jaccard 0.990 at initialization versus
0.317 warmed), because convergence moves the buffers toward the population rather than toward
this batch. What this probe establishes is the **sign and the isolation**, not a magnitude.

### 2.3 Branch normalization prevents a layer from attenuating itself (probe 3)

`||branch|| / ||residual stream||` entering each GINE / GINE_GSA layer, in train mode, as the
input magnitude changes:

| `normalization` | scale 0.1 | scale 1 | scale 10 |
|---|---|---|---|
| batch | [7.656, 0.991] | [1.212, 0.770] | [0.122, 0.122] |
| layer | [7.692, 0.986] | [1.214, 0.776] | [0.122, 0.123] |
| graph | [7.660, 0.991] | [1.213, 0.771] | [0.122, 0.122] |
| none | [1.438, 0.922] | [0.342, 0.369] | [0.255, 0.281] |

`batch`, `layer` and `graph` are indistinguishable and track 1/scale exactly: the branch
magnitude is pinned regardless of what the residual stream carries, so a layer cannot learn to
contribute less. Only `none` breaks the proportionality. GINE and GINE_GSA are identical.

**This corrects a claim made earlier in this session.** The effect follows from normalizing the
branch before the residual add
([nn_blocks.py:846](../../src/fragnnet/model/nn_blocks.py#L846),
[nn_blocks.py:1250](../../src/fragnnet/model/nn_blocks.py#L1250)) and is *not* specific to
BatchNorm. Switching `batch` to `layer` or `graph` does not address it. Only pre-norm plus a
learned branch gate would; the intensity head's `component_residual_head` and
`mask_residual_head` are both zero-initialized, so that pattern is already in the repository.

With `mol_num_layers: 2` in 205 of 226 runs this is survivable; it is a plausible reason the
4- and 5-layer runs (19 runs) did not pay off.

### 2.4 Context normalization is off everywhere

`mass_shatter_context_normalization` is never `layer` in any of the 226 runs, although the
projector comment at [mass_shatter_model.py:603](../../src/fragnnet/model/mass_shatter_model.py#L603)
explicitly flags that its input scale varies with the number of CE readings a merged
acquisition group contributed (median 4, max 86). One 128-d context vector is broadcast to
every bond, atom and candidate token of a molecule, so this is the model's highest-leverage
and least-regularized path. Note that the wired `mass_shatter_context_dropout` is unit dropout
*inside* the projector MLP ([mass_shatter_model.py:610](../../src/fragnnet/model/mass_shatter_model.py#L610));
whole-vector metadata dropout, the variant that would test reliance on acquisition metadata,
is not implemented.

### 2.5 Normalization outside the GNN (78% of parameters)

`mol_normalization` governs only the two encoders. Everything else, measured on the
production-shaped model after the atom decoder was removed (3.5, probe 9):

| Module | Parameters | Share | Norm layers | Type |
|---|---:|---:|---:|---|
| `intensity_head` | 8,835,092 | 53.2% | 13 | LayerNorm |
| `bond_break_head` | 3,865,473 | 23.3% | 7 | LayerNorm |
| `precursor_encoder` | 1,808,256 | 10.9% | 4 | `mol_normalization` |
| `proposal_encoder` | 1,808,256 | 10.9% | 4 | `mol_normalization` |
| `charge_head` | 263,169 | 1.6% | **0** | none |
| `context_encoder` | 23,206 | 0.1% | **0** | none by default |
| **total** | **16,603,452** | | | |

Removing the atom decoder took the model from 20,549,181 parameters and raised the GNN's share
from 17.6% to **21.8%** and the bond head's from 18.8% to **23.3%**, so it makes the BatchNorm
result in 2.1-2.2 more consequential, not less.

Three observations.

**The bond head already does the residual the right way.** Its update is `norm(x + f(x))`,
where the branch is *not* normalized, so unlike the GNN (2.3) it can attenuate a layer. Measured
`||branch|| / ||residual||` across three input scales:

| input scale | bond head | GNN |
|---:|---|---|
| 0.1 | [1.617, 0.169, 0.161] | [7.656, 0.991] |
| 1 | [0.224, 0.169, 0.166] | [1.212, 0.770] |
| 10 | [0.162, 0.168, 0.166] | [0.122, 0.122] |

The bond head converges to a stable ~0.17 regardless of input scale; the GNN tracks 1/scale.
The correct pattern therefore already exists in this repository, two files away, which makes
the GNN fix a local change rather than a new design.

**`charge_head` and the bond head's final scorer carry no normalization at all.** Both are bare
`_build_head` MLPs (Linear, SiLU, Dropout, Linear) consuming
`concat(normalized_embedding, unnormalized_context)`.

**The `context_encoder` has no normalization, and the source comment explaining why it might
need one is wrong.** [mass_shatter_model.py:603](../../src/fragnnet/model/mass_shatter_model.py#L603)
says the projector's input scale "varies with how many CE readings a merged acquisition group
contributed (median 4, max 86)". Measured (probe 10), it does not — both CE paths mean-pool, so
magnitude is flat:

| readings per spectrum | mean \|\|context\|\| | across-spectrum sd of \|\|context\|\| |
|---:|---:|---:|
| 1 | 2.1976 | 0.0155 |
| 4 | 2.2009 | 0.0116 |
| 22 | 2.2031 | 0.0052 |
| 86 | 2.2046 | 0.0018 |

Magnitude moves 0.3% from 1 to 86 readings. What actually changes is the *across-spectrum
spread*, which falls 8.6x, because averaging more CE readings pulls each spectrum's mean CE
toward the global mean. That is an expected consequence of mean-pooling, and since the merged
target is itself a consensus over those CEs, conditioning on the mean is arguably correct. It
is also **not** something `context_normalization: layer` repairs: LayerNorm fixes
`||context||` to a constant (across-spectrum sd exactly 0.0000) but cannot restore lost CE
variance. Treat the `ctxnorm` arm as scale equalization, not as a fix for this comment.

## 3. Generalization and optimization

### 3.1 `lr_schedule_type: cosine` is silently ignored — still live (probe 6)

`SpectrumPL.configure_optimizers`
([spectrum_pl.py:1766-1771](../../src/fragnnet/pl_model/spectrum_pl.py#L1766-L1771)) calls
`build_lr_scheduler` without `schedule_type`, `total_steps` or `min_lr_ratio`, so it takes the
`"exponential"` default. `FragnnetPL` forwards all three
([fragnnet_pl.py:1530-1537](../../src/fragnnet/pl_model/fragnnet_pl.py#L1530-L1537)).
Asking for cosine with `lr_total_steps: 100000` and `lr_min_ratio: 0.01`:

| step | MASShatter actually gets | cosine as configured |
|---:|---:|---:|
| 10,000 | 9.500e-04 | 9.799e-04 |
| 50,000 | 6.302e-04 | 5.129e-04 |
| 100,000 | **3.774e-04** | **1.000e-05** |

A 38x discrepancy at the end of training. This was recorded in the
[unmerged overfitting audit](masshatter_unmerged_overfitting_2026_09_16.md) on 2026-09-16 and
is unchanged in the working tree. That audit's own "next discriminating measurement" is an
LR-decay arm, so this must be fixed first or that experiment measures the wrong schedule.

### 3.2 Weight decay reaches normalization gains, biases and embeddings (probe 7)

`configure_optimizers` passes `self.parameters()` to AdamW with a single `weight_decay`, so
decoupled decay pulls LayerNorm/BatchNorm affine parameters, every bias, and the embedding
tables toward zero. On a production-shaped model:

```
trainable parameters            : 20,547,573
  normalization weight and bias :     26,112
  embedding tables              :         32
  remaining one-dimensional     :     44,444
  conventionally decay-exempt   :     70,588  (0.34%)
```

Small in count. The behavioural question is the constant pull on gains initialized at 1.0: at
lr 1e-3 and weight_decay 0.01, an unopposed parameter retains 0.905 of its value after 10,000
steps, 0.368 after 100,000, and 0.135 after 200,000. Gains do receive gradients, so these are
bounds on the pull rather than the trajectory — but with `lr_schedule: False` in 225 of 226
runs the pull never decays, and the merged leaders train 167-200 epochs at batch 64.

This is cheap hygiene (a no-decay parameter group), not a headline.

### 3.3 Constant learning rate, and why post-LN is *not* the reason to change it

`lr: 0.001` with no schedule and no warmup in 225 of 226 runs. The intensity stack is fully
post-LN — `_InducedCandidateAttentionLayer` and `_FullCandidateAttentionLayer` use
`norm(x + dropout(sublayer(x)))`, and the atom decoder is `nn.TransformerDecoderLayer` with
`norm_first=False` and `nn.TransformerDecoder(norm=None)` — which suggests the textbook
post-LN-needs-warmup argument.

**Measured, that argument does not hold here (probe 8).** Per-layer gradient norm at
initialization, under a realistic readout (linear head to a per-candidate logit, cross-entropy
over a molecule's candidates), against a pre-norm twin with identical parameter count:

| depth | post-LN, as built (max/min) | pre-LN twin (max/min) |
|---:|---:|---:|
| 2 | 1.02 | 1.04 |
| 3 | 1.01 | 1.05 |
| 6 | 1.04 | 1.13 |
| 12 | 1.07 | 1.23 |

The post-LN stack is *flat* at every depth, and marginally flatter than pre-LN. Production uses
depth 2 (90 configs) or 3 (56 configs), far from where post-LN pathology appears. **Do not
justify a warmup experiment by the post-LN structure.** The case for an LR schedule rests on
1.2 and 3.1 alone: it has never been tested with a matched control, and the one run that tried
it is unusable.

An earlier version of this probe applied an `x.pow(2)` loss directly to the stack output. Because
post-LN ends in a LayerNorm, that loss is nearly scale-invariant upstream and produced a spurious
~4e5 gradient ratio. Use a readout and a real loss.

That stack is nonetheless not a peripheral module: after the atom decoder's removal the
post-LN candidate-attention `layers` are 8,284,416 parameters, 93.8% of the intensity head and
**49.9% of the whole model**. The full breakdown is in 2.5.

### 3.5 The atom decoder was removed (user decision, 2026-09-19; implemented)

Supporting evidence from the `..._warmup_e0_4_dense_subformula_fast` family, which differs only
in `mass_shatter_intensity_atom_decoder_num_layers`, all at batch 64 for 200 epochs:

| Run | decoder | cosine @100 | @150 | @166 | best | wall clock | min/epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| [mgs5qts9](https://wandb.ai/frag-gnn/frag-gnn/runs/mgs5qts9) | 4 | 0.7925 | 0.7920 | 0.7962 | 0.7990 | 8.7 h | 2.6 |
| [lzhb3ozh](https://wandb.ai/frag-gnn/frag-gnn/runs/lzhb3ozh) | 4 | 0.7894 | 0.7898 | 0.7899 | 0.7941 | 8.7 h | 2.6 |
| [6ru0lg6f](https://wandb.ai/frag-gnn/frag-gnn/runs/6ru0lg6f) | 0 | 0.7916 | 0.7968 | 0.7950 | 0.7981 | 4.9 h | 1.5 |
| [zp06g7ml](https://wandb.ai/frag-gnn/frag-gnn/runs/zp06g7ml) | 0 | 0.7930 | 0.7946 | 0.7947 | 0.7987 | 4.8 h | 1.4 |

Epoch-matched at 166 the two-run means are 0.7931 with the decoder and 0.7949 without, a
difference of −0.0018 in the decoder's favour-of-removal direction and well inside the ±0.006
to 0.012 replicate floor from 3.4. Clock-matched it is not close: **1.8x faster per epoch**, and
the decoder-free runs reach 0.7987 in 4.8 h against 0.7990 in 8.7 h. The decoder buys no
measurable cosine for 19.2% of parameters and 45% of the wall clock.

Two practical consequences:

- **`config/models_template` already defaults `mass_shatter_intensity_atom_decoder_num_layers: 0`**,
  and 104 of 156 configs inherit it. Only 51 configs set 4 explicitly. The `ma_mi_merged`
  normalization arm and its 4x-replicated baseline (3.4) are already decoder-free, so the
  recommended `graph` experiment needs no config change.
- **The decoder code path was deleted in this session** (`_refine_from_fragment_atoms`,
  `atom_projection`, `atom_decoder_gate`, `atom_decoder`, and the `_MAX_ATTENTION_SEQUENCES`
  CUDA-grid chunking that existed only to serve it), along with the head's three
  `atom_*` forward arguments. Net -132 lines under `src/`.
- Two guards replace it. `MassShatterPL` raises at setup if
  `mass_shatter_intensity_atom_decoder_num_layers` is nonzero, and
  `CandidateIntensityTransformerHead._load_from_state_dict` raises if a checkpoint carries
  `atom_decoder.*` or `atom_projection.*` weights, so a decoder-trained checkpoint fails
  loudly instead of silently serving a different model. The config key is retained in
  `config/models_template` at 0 because `load_config` asserts every custom-config key exists
  there; 51 configs that set 4 now omit it and inherit 0, and 45 stale
  `intensity_atom_decoder_layers_4` W&B tags were dropped.

## Corrections to claims made earlier in this session

- **Residual-branch pinning is not BatchNorm-specific.** `layer` and `graph` behave identically
  (2.3). An earlier statement implied switching normalization would help; measurement says it
  would not.
- **Dropout is not the main driver of candidate-set train/eval skew.** BatchNorm is, and its
  contribution is deterministic (2.2). The initial review attributed the skew to dropout alone.
- **"Large molecules dominate the batch statistic"** is now measured rather than asserted:
  0.7900 relative movement when companions grow 30 to 120 atoms (2.1).
- **The post-LN warmup argument does not survive measurement.** The initial review claimed the
  fully post-LN intensity stack "wants warmup". Measured gradient profiles are flat at depths
  2, 3, 6 and 12, and marginally flatter than a pre-LN twin (3.3). The case for an LR schedule
  rests on the absence of a matched control, not on post-LN.
- **The source comment about context scale is wrong** and the initial review repeated it. The
  context vector's magnitude does not vary with the CE-reading count (2.5).
- A discarded probe attempted to emulate a trained model by rescaling the bond head's output
  layer. That scales signal and dropout noise equally and is a no-op on ranking; it produced
  identical Jaccard at every scale and proves nothing. Do not repeat it.

## Limits of this evidence

- Every probe uses randomly initialized weights. A trained model's weights adapt to train-mode
  statistics, which plausibly shrinks the BatchNorm gap; nothing here bounds by how much.
- The candidate-set probe uses a two-molecule batch, which maximizes the batch-versus-population
  statistic gap. Real training batches are 32-512.
- BatchNorm warm-up uses synthetic random node features, not the NIST20 atom distribution.
- W&B `config` reflects the launch configuration, not necessarily the code revision that ran.
- No claim here is a measured improvement. Nothing was trained.

### 3.4 A noise floor for any normalization arm

`nist20_ma_mi_merged_fast_batch512_cuda_lb64` was launched four times on 2026-09-18, all with
`seed: 397623`, all crashed on wall time at different epochs. Same config, same seed, so the
spread below is pure run-to-run nondeterminism (CUDA candidate backend, scatter atomics,
dataloader workers), not seed variance.

| epoch | iep3gc9f | o7zpxi78 | bpbome2x | zzm0p7dv | spread | sd |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.6647 | 0.6697 | 0.6719 | 0.6595 | 0.0124 | 0.0055 |
| 20 | 0.6921 | 0.6972 | 0.6939 | 0.6911 | 0.0061 | 0.0027 |
| 25 | 0.7024 | 0.7016 | 0.6949 | 0.6975 | 0.0075 | 0.0035 |
| 29 | 0.6951 | 0.7017 | 0.7035 | 0.6969 | 0.0084 | 0.0040 |
| 36 | 0.7091 | — | 0.7060 | 0.7046 | 0.0045 | 0.0023 |

**Decision rule: a single normalization or regularization arm must beat its matched baseline by
more than roughly 0.01 validation cosine at a matched epoch to mean anything**, and smaller
differences need replication. This retrospectively confirms 1.2: the layer-normalization run's
differences from its siblings (0.0025 and 0.0099) sit inside this floor, so that comparison
could not have concluded anything even had the batch sizes matched.

Two useful consequences. First, this family already has a 4x-replicated `batch` baseline, so a
`graph` or `layer` arm launched from `config/massshatter/ma_mi_merged/` gets a matched control
for free. Second, `..._lb64_graphnorm.yml` states in its own header that only
`mol_normalization` and `wandb_name` differ from `..._lb64.yml`, and it carries the current
`merge_method: tolerance_consensus` target, so it is a valid one-variable arm today.

## Next discriminating measurement

Ordered by dependency, not by expected effect size:

1. **Fix the scheduler forwarding** in `SpectrumPL.configure_optimizers` so `lr_schedule_type`,
   `lr_total_steps` and `lr_min_ratio` reach `build_lr_scheduler`. Prerequisite for 3, not an
   experiment.
2. **Measure the BatchNorm candidate-set gap on a trained checkpoint**, in the two modes, on a
   fixed validation subset. This converts 2.2 from a mechanism into a magnitude and is the only
   step that needs no new training. If the gap is material, `mol_normalization: graph` becomes
   a correctness fix rather than a hyperparameter, and `..._lb64_graphnorm.yml` already exists
   with a 4x-replicated matched baseline (3.4).
3. **One matched LR arm** (warmup plus decay versus constant) at equal optimizer budget and
   equal batch size, which `qz0uelf0` did not have.
4. **The four remaining prepared ablations** (`drop02`, `drop03`, `wd10`, `ctxnorm`), which are
   written and have never been launched.

Pre-norm plus a zero-initialized branch gate in GINE/GINE_GSA (2.3) is a model-code change and
should follow 2, not precede it: if candidate selection is the dominant defect, changing the
encoder's residual structure would confound the measurement.
