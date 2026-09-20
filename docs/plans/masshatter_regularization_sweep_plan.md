# MASShatter regularization and normalization sweep (Fir, one-hour boxes)

> **Status:** PROPOSED · **Updated:** 2026-09-19
> Nine sweep configs written and verified locally (eight arms plus the 2g slice variant); replicate
> floor, throughput and the GPU-memory / slice-sizing table measured from existing W&B runs.
> **Stage A was submitted on Fir on 2026-09-19 and crashed**; the cause is fixed in
> `spectrum_pl.py` (see [Stage A incident](#stage-a-incident-2026-09-19--out-of-range-predicted-peaks))
> and Stage A must be resubmitted from a checkout carrying that fix.

Executes the "sweep second" half of the
[regularization audit](../findings/masshatter_regularization_audit_2026_09_19.md), whose census
established that `mass_shatter_head_dropout`, `weight_decay`, `lr` and `gradient_clip_val` are
identical in all 226 (now 232) `model_type: mass_shatter` runs, and that
[the overfit audit's](../findings/masshatter_unmerged_overfitting_2026_09_16.md) instrument gate
is still closed.

## Objective

Screen, at direction-finding resolution, whether any single regularization or normalization knob
moves the merged MA/MI arm off its current trajectory — and, for the first time in this model
family, observe a **train** cosine next to the validation cosine while doing it.

This is a screen, not a decision. It is designed to rank eight one-knob arms and produce one or two
candidates worth a multi-hour chained run, not to select a production setting.

### Acceptance criteria

| # | Criterion | How it is read |
| --- | --- | --- |
| A1 | `train_cos_sim_0.01_epoch/mean` is logged on every arm | W&B key exists; this has never been true for any MASShatter run |
| A2 | Every arm reaches a common epoch `E*` >= 20 inside its one-hour box | per-arm `epoch` summary |
| A6 | The 2g-to-3g `min/epoch` ratio is measured before the fan-out, and the sweep shape follows it | Stage A pilot pair |
| A3 | The two controls agree within the measured replicate spread (<= 0.007 val cosine at `E*`) | if they do not, the screen is underpowered and no arm is promoted |
| A4 | Each arm is classified promote / hold / reject against the control mean at `E*` | see [Read-out](#4-read-out-protocol) |
| A5 | Resolved backend recorded as `cuda` on every run | `mass_shatter_candidate_backend_resolved` W&B summary field |

## 1. Established facts

All measured this session unless linked elsewhere.

### The replicate floor is ~0.003, so the screen resolves ~0.007

Four runs of the **identical** base config (`nist20_ma_mi_merged_fast_batch512_cuda_lb64`) on
Vulcan L40S, compared at matched epoch:

| Run | ep 10 | ep 15 | ep 20 | ep 25 | ep 28 | epochs reached |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `iep3gc9f` | 0.6647 | 0.6846 | 0.6921 | 0.7024 | 0.7047 | 37 |
| `o7zpxi78` | 0.6697 | 0.6816 | 0.6972 | 0.7016 | 0.6975 | 30 |
| `bpbome2x` | 0.6719 | 0.6895 | 0.6939 | 0.6949 | 0.7011 | 47 |
| `zzm0p7dv` | 0.6595 | 0.6857 | 0.6911 | 0.6975 | 0.7003 | 56 |
| **spread** | 0.0124 | 0.0079 | 0.0061 | 0.0075 | 0.0072 | |

`val_cos_sim_0.01_epoch/mean`. Replicate sd is about **0.003** from epoch 20 on, spread about
0.007. A single one-hour arm therefore resolves effects of roughly **+/-0.007 or larger**. Eight arms
at n=1 will produce one false 0.007-level separation on average; that is why the read-out below
promotes on mechanism plus margin, never on margin alone.

### "Best cosine at one hour" is not a valid comparator

The same config ran at **2.37 min/epoch** (`iep3gc9f`, 2026-09-18 05:52) and **0.95 min/epoch**
(`zzm0p7dv`, same day 14:20) — 2.5x, from the batched proposal-coverage loss landing in between
(see the [training-speed finding](../findings/masshatter_training_speed_2026_09_17.md#batched-proposal-coverage-loss-2026-09-18)).
Epoch counts inside a
fixed hour ranged 30-56 across those four runs. **Every comparison in this plan is at matched
epoch.** Wall-clock is a budget here, not a variable.

### Slice sizing is a measured open question, not a settled choice

NVML peak device memory across every merged MA/MI run (GiB, since MIG quotes GiB):

| Arm | GPU peak | Fits `2g.20gb` (~19.5 GiB usable)? |
| --- | ---: | --- |
| 128 cuda lb8 (`qhs040sy`) | 4.9 GiB | yes |
| 256 cuda lb8 (`r7bavpb0`) | 8.0 GiB | yes |
| **256 cuda lb32** (`0sfedmop`) — the 2g shape | **12.7 GiB** | **yes, ~6.8 GiB margin** |
| 512 cuda lb8 (`aqm8u9p4`) | 14.1 GiB | yes |
| *512 cuda lb16 (interpolated)* | *~15.8 GiB* | *yes, but projected, not measured* |
| 256 cuda lb64 (`zxlj1iru`) | 17.3 GiB | yes, 2.2 GiB margin |
| 512 cuda lb32 (`arnuiim5`, `ewpumd0d`) | 18.7-19.3 GiB | **too tight** |
| **512 cuda lb64** (this sweep's base) | **25.1-26.9 GiB** | no |

Memory tracks batch size x `loss_batch_size` at about 0.22 GiB per lb unit above an ~12.3 GiB base.

**`loss_batch_size` is not a modeling knob.** `_build_loss_chunks` (`loss.py:629-675`) walks the
batch in windows of at most `loss_batch_size` spectra, subdividing further on a padding heuristic,
and reduces per-spectrum losses; changing it alters float accumulation order and nothing else. So
**batch 512 can be kept on a 20 GB slice by running `lb16`.** Earlier drafts of this plan claimed a
smaller `lb` would break continuity with the `512 x lb64` replicates. That was wrong. The 2g shape
nevertheless uses microbatch 256 with `accumulate_grad_batches: 2` instead, by user decision: it
rests on a measured 12.7 GiB rather than an interpolated 15.8 GiB, and keeps the effective batch at
512. The cost is the BatchNorm interaction noted in [Stage A](#stage-a--paired-slice-pilot-2-jobs-1-h-wall-clock-958-units).

What does not cancel is MIG's non-proportional slicing. `3g.40gb` carries **four** of the eight
memory slices, not three:

| Profile | Compute | Memory slices | CPUs | Price | Bandwidth per unit price |
| --- | --- | --- | ---: | ---: | ---: |
| `fir:h100-2g.20gb` | 2/7 | **2/8** | 3 | 3.48 | 0.0718 |
| `fir:h100-3g.40gb` | 3/7 | **4/8** | 6 | 6.10 | **0.0820** |

Dropping to 2g costs a third of the compute but **half the bandwidth**, against a 1.75x price
advantage. On `zzm0p7dv`, NVML memory utilization runs mean 72% / p50 90% — a measure of how often
memory is accessed, not of peak GB/s, so it is suggestive rather than conclusive, but it points at a
1.5-2x slowdown that would consume the price advantage.

Per-job fixed overhead is small and does not decide this: in-run setup to the end of epoch 0 is
2.6 min and the job occupies 57 of 60 box-minutes, so about 6 min (10%) of each one-hour box.

The train-side eval tensors added by `compute_train_auxiliary_metrics` are built with `sparse=True`
(`spectrum_pl.py:619-629`), so they are peak-list sized (~27k peaks at batch 512, merged rows
averaging 51.9 peaks), well under a GiB. The instrumentation costs **time, not memory**.

### Equal-budget shapes, decided by the Stage A measurement

At roughly 49 price units:

| | Profile / config | Jobs | Epochs per arm | Comparator resolution |
| --- | --- | --- | ---: | --- |
| **A** | `3g.40gb`, `512 x lb64` | 8 x 1 h | ~55 | arm n=1 vs control n=2, se 0.0037 -> ~0.0073 |
| **B** | `2g.20gb`, `256 x 2 x lb32` | 14 x 1 h | ~38 at 1.5x, ~28 at 2x | arm n=2 vs control n=2, se 0.003 -> ~0.006 |
| **C** | `2g.20gb`, `256 x 2 x lb32` | 8 arms x 2 h chained | ~76 at 1.5x, ~56 at 2x | as A, with more exposure |

B buys a modest threshold improvement by spending 30-50% of the epochs, and thin exposure is
already what handicaps the dropout arms (see [the bias note](#the-screen-is-biased-against-the-dropout-arms)).
C is attractive only if the 2g slowdown is nearer 1.5x than 2x. **Stage A measures the ratio
instead of assuming it.**

### The arms exist, resolve, and are one knob each

All eight configs under `config/massshatter/ma_mi_merged/` resolve through `load_config` against
the current `config/models_template`, and the resolved diff against the base is exactly one key:

| Arm config suffix | Resolved delta from base | Prior |
| --- | --- | --- |
| *(base)* | — | control |
| `_layernorm` | `mol_normalization: batch -> layer` | **highest**: at dropout 0, train-vs-eval Jaccard of the selected m/z set is 0.317/0.611 under `batch` and exactly **1.000** under `layer` |
| `_graphnorm` | `mol_normalization: batch -> graph` | same mechanism, per-graph statistics, also Jaccard 1.000 |
| `_drop02` | the three live dropouts `0.1 -> 0.2` | direct test of the untested "overfits but not under-regularized" claim |
| `_drop03` | the three live dropouts `0.1 -> 0.3` | upper bracket of the pair |
| `_wd10` | `weight_decay: 0.01 -> 0.1` | separates decay from dropout; identical in 232/232 runs |
| `_ctxnorm` | `mass_shatter_context_normalization: none -> layer` | scale equalization on the metadata projector that conditions the GNN and every head |

The three dropout keys are wired at `mass_shatter_pl.py:966/974/993`; `context_normalization` at
`mass_shatter_pl.py:964`; `graph` normalization at `nn_blocks.py:297,1431`.

No arm sets `mass_shatter_intensity_atom_decoder_num_layers`, so all eight run the post-removal
fast path (`a55b0c7f`) and no arm trips the new setup guard.

### Two knobs the audits list are not available here

- **`cc_atom_dropout` / `cc_feature_dropout` / `cc_channel_dropout` are FraGNNet-only.** They are
  read at `fragnnet_pl.py:170-172` and consumed by `FragnnetModel`; nothing in the MASShatter path
  reads them. "0 in 232/232 mass_shatter runs" is true and **not actionable** — they are inert
  there. This corrects the implication in the overfit audit's unswept-axes list.
- **No LR-schedule arm.** `SpectrumPL.configure_optimizers` (`spectrum_pl.py:1766`) still calls
  `build_lr_scheduler` with only `decay_rate`/`decay_steps`/`warmup_steps`, dropping
  `lr_schedule_type`, `lr_total_steps` and `lr_min_ratio` that `utils/nn_utils.py:22` accepts. A
  `cosine` arm would silently run exponential, as it did in six earlier runs. Fix that forwarding
  in its own change, with its own test, before any LR arm.

## 2. The sweep

Eight one-knob arms plus two controls, in one-hour boxes. The GPU profile and the number of
replicates per arm are **set by the Stage A slice measurement**, not fixed here: shape A (10 jobs on
`3g.40gb`) if the 2g slice is >= 1.75x slower, shape B (18 jobs on `2g.20gb`, n=2 per arm) if it is
<= 1.5x. Job names below are shape A; shape B appends `_s2` for the second seed of each arm.

All eight configs exist under `config/massshatter/ma_mi_merged/`, sharing the stem
`nist20_ma_mi_merged_fast_batch512_cuda_lb64`. The experiment is **in the files**, not in launch
flags: each carries `compute_train_auxiliary_metrics: true`, `wandb_group:
nist20_mass_shatter_reg_sweep` and the tags `reg_sweep_2026_09_19` / `train_aux_metrics`, so a
missing override cannot silently break comparability, and the W&B read-out is one group query.

| Job name | Config suffix | Axis moved vs `ctrl_a` |
| --- | --- | --- |
| `ms_reg_ctrl_a` | `_ctrl_a` | — (reference) |
| `ms_reg_ctrl_b` | `_ctrl_b` | `seed: 397623 -> 811923` |
| `ms_reg_layernorm` | `_layernorm` | `mol_normalization` |
| `ms_reg_graphnorm` | `_graphnorm` | `mol_normalization` |
| `ms_reg_drop02` | `_drop02` | the three dropout sites, together |
| `ms_reg_drop03` | `_drop03` | the three dropout sites, together |
| `ms_reg_wd10` | `_wd10` | `weight_decay` |
| `ms_reg_ctxnorm` | `_ctxnorm` | `mass_shatter_context_normalization` |
| `ms_reg_hbound_complement` | `_hbound_complement` | `mass_shatter_h_transfer_bound` |
| `ms_reg_hbound_valence` | `_hbound_valence` | `mass_shatter_h_transfer_bound` |

Verified after generation: each config resolves against `config/models_template`, has the identical key
set, and differs from `ctrl_a` in **exactly** the keys of its declared axis and nothing else. The
original `nist20_ma_mi_merged_fast_batch512_cuda_lb64.yml` is left untouched (still
`compute_train_auxiliary_metrics: False`) so the four-replicate lineage keeps its config identity;
`_ctrl_a` is its instrumented copy. The six arm files were instrumented in place — none had ever
been run, so there was no history to preserve and no reason to duplicate them.

**The two dropout arms move three keys, not one.** `mol_dropout`, `mass_shatter_head_dropout` and
`mass_shatter_intensity_transformer_dropout` shift together, as the prepared configs were written.
That is one *axis* but not one *site*: if the GNN and the head want different amounts, this arm
cannot say so, and a null here does not exonerate dropout site by site. Accepted for a screen;
a positive result should be followed by a per-site split, not shipped as-is.

Two controls, not one: the four-run floor above was measured on L40S under pre-2026-09-18-afternoon
code, and this sweep changes GPU, code and instrumentation at once. The second control re-derives
the floor **inside** the sweep, on the same hardware and the same instrumented code path, which is
what A3 checks. It is the cheapest insurance available against reading a 0.007 artifact as a result.

`compute_train_auxiliary_metrics` is in **all eight** configs, including the controls, so throughput
and metric definitions stay uniform. It changes the epoch rate relative to the existing L40S runs;
those runs are already not clock-comparable, so nothing is lost.

`loss_batch_size` and `num_workers` are deliberately **not** in the sweep configs: they are slice
properties, not experiment settings, and the Stage A measurement may move them. They are passed at
launch, where they appear in the generated job script and the W&B config.

**The two hydrogen-transfer arms are not regularization arms and are not read like one.** They
change the candidate set rather than the optimizer, and they remove the only channel currently
explaining M+1 isotope peaks, so roughly 0.2% of intensity weight moves to unexplained and the
primary loss shifts for a reason unrelated to fit quality. **Compare them on val cosine at matched
epoch, never on the loss**, and read them against `ctrl_a`/`ctrl_b` like everything else. A null
result is informative in its own right: it would say the unreachable channels are neither helping
as softmax negatives nor hurting as isotope crutches, which settles whether the bound can simply be
turned on. See [the incident section](#stage-a-incident-2026-09-19--out-of-range-predicted-peaks)
for the mode definitions and the measured candidate counts.

`_warmup_e0_4` is deliberately excluded — it is staged training, not regularization, and it moves
the optimization schedule rather than the capacity/normalization axis under study here.

## 3. Execution

### Stage A — paired slice pilot (2 jobs, ~1 h wall clock, 9.58 units)

Submit the control config **twice, in parallel, on both slices**, and let both run the full hour.
Both are real control arms, not throwaways — whichever slice wins, its pilot becomes `ctrl_a` for
the sweep and the other is kept as cross-slice calibration.

| Pilot | Profile | Config |
| --- | --- | --- |
| `ms_reg_ctrl_a` | `fir:h100-3g.40gb` | `..._batch512_cuda_lb64_ctrl_a.yml` |
| `ms_reg_ctrl_2g` | `fir:h100-2g.20gb` | `..._batch256x2_cuda_lb32_ctrl_2g.yml` |

Neither needs a launch override. The 2g config drops the **microbatch** to 256 with
`accumulate_grad_batches: 2`, so the effective batch stays 512 and the optimizer math of the lb64
lineage is preserved; `loss_batch_size` follows to 32 and `num_workers` to 3 for that profile's 3
CPUs. Measured peak for `256 x lb32` is **12.7 GiB** (`0sfedmop`) against ~19.5 GiB usable — about
35% headroom, which the `512 x lb16` alternative (~15.8 GiB, and only ever interpolated) did not
have. The pair measures the one number this plan cannot derive: **`min/epoch` on 2g divided by
`min/epoch` on 3g**, which decides between shapes A, B and C above.

**The microbatch change lands on the axis under study.** BatchNorm statistics are computed per
microbatch, so `mol_normalization: batch` at 256 is not the same operator as at 512 — and that is
precisely what `_layernorm` and `_graphnorm` are testing against. A uniform microbatch across all
arms keeps the sweep internally valid, but if the 2g shape is selected: every arm must carry the
same five keys, effect sizes do not transfer to a 512-microbatch run, and nothing in that shape may
be read against the four `512 x lb64` L40S replicates. This is a further reason to prefer 3g unless
the measured slice ratio is clearly favourable.

Gate the fan-out on:

1. The native CUDA selection/topology libraries **build for sm_90**. Every prior run of this config
   is L40S (sm_89); this is the first H100 build. The source compiles per-capability into a `/tmp`
   cache (`mass_shatter_cuda_selection.py:63-92`), which on Fir is node-local, so budget one build
   (180 s timeout) per job, not once per campaign.
2. `mass_shatter_candidate_backend_resolved == cuda` (A5). Saved run config can overwrite local
   settings on resume; read it from the run, not from the file.
3. `train_cos_sim_0.01_epoch/mean` appears in the history (A1).
4. No OOM on either slice with train-side eval tensors live, and the **measured** 2g peak against
   the 15.8 GiB projection. If 2g OOMs, retry once at `loss_batch_size=8` (14.1 GiB measured) before
   discarding the 2g shape.
5. Record min/epoch and epochs/hour on both. At the L40S rate of 0.95 min/epoch a 3g box should
   reach ~50 epochs after its ~6 min of overhead; anything under ~20 epochs (A2) on the chosen
   slice means the box is too small and the sweep moves to two-hour boxes before fanning out.
6. **The slice ratio decides the shape.** <= 1.5x -> shape B or C on 2g; >= 1.75x -> shape A on 3g;
   in between, prefer 3g, because epochs are the scarcer resource for the dropout arms.

### Stage A incident, 2026-09-19 — out-of-range predicted peaks

The first Stage A submission (SLURM `60545834`) died inside `training_step` at
`batched_bin_func`: `AssertionError: (tensor(1501.7432), 1500)`.

**Cause: two bounds that were never reconciled.** `mz_max` bounds the *observed* spectrum — target
peaks pass through `rel_filter_func` with it (`spectrum_pl.py:296-306`), so they are always in
range. The *candidate grid* is bounded by something else entirely: each candidate carries an
H-transfer ladder of `mass_shatter_max_h_transfer: 4`, so the intact-molecule candidate reaches
`prec_mz + 4 x 1.00783`. Predicted peaks are never range-filtered, because `preproc_spec`'s
`(not train) and pred` branch (`spectrum_pl.py:1234-1238`) only untransforms and normalizes
**intensities**. `batched_bin_func` then asserts `max(mzs) < mz_max` on both sides.

**Measured on this split.** The heaviest training precursor is **1497.7119**, and
1497.7119 + 4 x 1.00783 = **1501.7432** — the assertion value, digit for digit. So the offending
peak is not a precursor and not a bad prediction: it is the top rung of the H ladder on the intact
molecule. The collision window is only 4.03 Da wide, so it catches exactly the molecules with
`prec_mz > 1495.97`:

| Fold | max `prec_mz` | spectra at risk | molecules at risk |
| --- | ---: | ---: | ---: |
| train | 1497.71 | **2** | 2 |
| val | 1477.68 | 0 | 0 |
| test | 1449.65 | 0 | 0 |

Two training spectra were enough to kill the job. **Correction to an earlier claim in this plan:
validation could not have tripped this on `nist20_ma_mi_inchikey`** — its val and test folds have
no molecule in the window at all. The code path is genuinely unguarded, but whether it fires is
split-dependent, so this was not a near-miss in the 232 historical runs; it is a hazard for any
split whose evaluation fold carries a precursor within 4 H of `mz_max`.

Worth a separate look, not fixed here: those `+nH`-above-precursor channels are physically
unreachable outcomes, and the model still spends probability mass on them. Bounding the H ladder
by `mz_max` at candidate construction would remove both the waste and the crash at the source; that
is a modeling change, and this sweep is not the place for it.

**Root-cause fix, behind a switch.** MASShatter's only channel rule was `crf_counts >= 0`, which
bounds hydrogen *loss* but never *gain*: a fragment could acquire hydrogens that exist nowhere, and
the intact molecule reached `prec_mz + 4H`. FraGNNet has always bounded this
(`frag_utils.py:518-533`). The bound is now a config option, `mass_shatter_h_transfer_bound`,
**defaulting to `none` so no existing run changes**:

| Mode | Rule | FraGNNet term |
| --- | --- | --- |
| `none` (default) | element counts nonnegative only -- the historical ladder | — |
| `complement` | a gain may not exceed the complementary neutral fragment's hydrogens; the intact molecule can only lose them | the `num_hs` cap |
| `valence` | also caps the gain at the fragment's free valence slots plus one per severed bond | `compute_cc_h_cap` |

`valence` is full parity: a lone O from one severed bond becomes OH, never OH3, however much
hydrogen the complement holds. It needs a new per-atom field, `atom_free_valence_slots`
(`feat_utils._atom_free_valence_slots`), computed with FraGNNet's own `ve`/`sbond` arithmetic and
its formal-charge correction. Molecule graphs are built per item with no disk cache, so nothing
stale needs regenerating. The unique-fragment path passes exact per-fragment boundary counts; the
component path passes the mask-level count, which over-counts an end piece of a 3-way cut and is
therefore conservative (it can leave an unreachable channel, never remove a reachable one).

**FraGNNet's own functions are reused as the test oracle, not at runtime.**
`compute_cc_h_cap` (numpy) and `compute_cc_h_floor` (numpy and Cython) each take one fragment as a
`cc_atom_ids` array, so calling them inside the batched GPU forward would need a host sync and a
Python loop over every fragment in the batch -- the cost the proposal-coverage batching removed
(118.6 ms -> 2.4 ms). MASShatter instead pre-aggregates the identical arithmetic per atom and adds
one slot per severed bond. `test_valence_bound_matches_fragnnet_reference` pins that identity by
running FraGNNet's `extract_mol_info` -> `update_bonds` -> `compute_cc_h_cap` as the reference for
every fragment of six molecules -- ethanol, aspirin, caffeine, acetonitrile (triple bond),
cyclohexane (ring) and a quaternary-ammonium zwitterion (the formal-charge correction path) -- and
they agree exactly. The floor term is still FraGNNet-only; MASShatter keeps its weaker
`counts >= 0` form.

`formula_utils.mass_shatter_has_valid_ion` takes the same setting so the dataset admissibility
filter cannot disagree with the model. Only hydrogen-free molecules can change verdict there --
**2 of 35,025** in NIST20 -- and correctly so: a molecule with no hydrogen cannot form `[M-H]-`.

**Why it is a switch and not simply on.** Measured across four probe molecules at
`max_h_transfer: 4`:

| Mode | Emitted candidates | vs `none` | Above the precursor |
| --- | ---: | ---: | ---: |
| `none` | 1,193 | — | 104 |
| `complement` | 1,042 | **-12.7%** | 0 |
| `valence` | 982 | **-17.7%** | 0 |

Those unreachable channels are not pure waste: they act as **negatives in the candidate softmax**,
and they are currently the only thing explaining M+1 isotope peaks. 2.35% of in-scope training
spectra carry a true peak on an above-precursor channel; for `+1H` (2.27% of spectra on its own)
**84.3% are closer to the M+1 ¹³C isotope than to a hydrogen transfer**. Turning the bound on makes
those peaks unexplained -- median 4% of base peak, 34.7% of them at or above 10%, roughly 0.2% of
intensity weight. That is arguably correct (isotopes should not be laundered through an impossible
channel) but it is a trade to measure, not assume, and isotope handling is its own problem.
The three modes are therefore a ready-made ablation axis.

**Defence in depth, kept** (`spectrum_pl.py`, `calculate_all_auxiliary_metrics`): predicted peaks at
or above `mz_max` are still dropped before binning. The two bounds answer different questions --
chemistry (can this ion exist?) versus observation window (can this ion be binned?) -- and a dataset
containing a molecule heavier than `mz_max` would still need the second. Peaks in range are
untouched, so no metric that did not previously crash changes value, and the range assert stays live
for the target side, where an out-of-range peak really does mean a broken pipeline.

**Fix** (`spectrum_pl.py`, `calculate_all_auxiliary_metrics`): drop predicted peaks at or above
`mz_max` before binning, so both sides are compared over the same `[0, mz_max)` window. Peaks in
range are untouched, so no metric that did not previously crash changes value, and the range assert
stays live for the target side, where an out-of-range peak really does mean a broken pipeline.
**Scope of the fix: the binned metrics only.** The edit sits inside `if compute_bin_specs:`, which
only `cos_sim` and `opt_cos_sim` enable (`spectrum_pl.py:396,430`). Not touched:
`recall`/`precision`/`wrecall`/`wprecision`, which go through `round_aggregate_peaks` and have no
range assert — an out-of-range prediction still counts there as unmatched, as it always has; and
the loss, which never bins for MASShatter (`self.binned_loss = False`, `mass_shatter_pl.py:1227`).
The binned-loss path (`_preproc_spec(bin_spec=...)` -> `batched_bin_func` at `spectrum_pl.py:323`)
carries the *same* unfiltered-prediction exposure and is **not** covered by this fix; it is dead for
MASShatter but live for `BinnedPL` (`binned_pl.py:57`) and for FraGNNet under a binned loss.

### Stage B — fan-out (8 or 16 jobs, ~1 h wall clock)

Submit the rest together on the slice Stage A selected: eight arms plus `ctrl_b` (shape A), or
sixteen arms plus `ctrl_b` (shape B, n=2 per arm). They are independent; one-hour boxes are chosen partly
because they schedule quickly, so many short jobs should clear faster than one long one.

### Stage C — chain the survivors (optional)

Only arms that clear the read-out get more time. `j=<name>` resumes from `last.ckpt`:

```bash
JOBID=$(sbatch --parsable slurm_scripts/ms_reg_<arm>.sh)
for _ in $(seq 1 3); do
  JOBID=$(sbatch --parsable --dependency=afterany:"$JOBID" slurm_scripts/ms_reg_<arm>.sh)
done
```

`afterany` is required because wall-clock termination is the normal exit for a capped job. Confirm
each link logs `resuming from checkpoint` rather than restarting at epoch 0, and that
`wandb/run-*/files/ckpt` in the Fir checkout holds `last.ckpt` between jobs.

### Generation commands

Run on the **Fir login node**, from the Fir checkout, with the cluster's `python` — not
`FRAGNNET_PYTHON`. Generates only; nothing is submitted. The generated scripts embed a resolved W&B
credential, so keep them private.

```bash
FRAGNNET_ACCOUNT=rrg-dwishart
FRAGNNET_EMAIL=fw4@ualberta.ca
CFG=config/massshatter/ma_mi_merged/nist20_ma_mi_merged_fast_batch512_cuda_lb64

# control A + the eight arms (shape A, 3g.40gb: no overrides needed at all)
for arm in _ctrl_a _layernorm _graphnorm _drop02 _drop03 _wd10 _ctxnorm \
           _hbound_complement _hbound_valence; do
  name="ms_reg${arm}"
  python scripts/setup_job_cc.py \
    --frag_gnn_dp . \
    --python_script_fp scripts/run_pl_model_fit.py \
    --job_name "$name" \
    --gpu_profile fir:h100-3g.40gb \
    --cc_account "$FRAGNNET_ACCOUNT" \
    --cc_email "$FRAGNNET_EMAIL" \
    --time 1:00:00 \
    --script_kwargs w=online c="${CFG}${arm}.yml" j="$name" \
    --no_submit=True
  bash -n "slurm_scripts/${name}.sh"
done

# control B: second seed, carried by its own config
python scripts/setup_job_cc.py \
  --frag_gnn_dp . \
  --python_script_fp scripts/run_pl_model_fit.py \
  --job_name ms_reg_ctrl_b \
  --gpu_profile fir:h100-3g.40gb \
  --cc_account "$FRAGNNET_ACCOUNT" \
  --cc_email "$FRAGNNET_EMAIL" \
  --time 1:00:00 \
  --script_kwargs w=online c="${CFG}_ctrl_b.yml" j=ms_reg_ctrl_b \
  --no_submit=True
bash -n slurm_scripts/ms_reg_ctrl_b.sh
```

The Stage A 2g pilot differs only in profile and two overrides:

```bash
python scripts/setup_job_cc.py \
  --frag_gnn_dp . \
  --python_script_fp scripts/run_pl_model_fit.py \
  --job_name ms_reg_ctrl_2g \
  --gpu_profile fir:h100-2g.20gb \
  --cc_account "$FRAGNNET_ACCOUNT" \
  --cc_email "$FRAGNNET_EMAIL" \
  --time 1:00:00 \
  --script_kwargs w=online j=ms_reg_ctrl_2g \
    c=config/massshatter/ma_mi_merged/nist20_ma_mi_merged_fast_batch256x2_cuda_lb32_ctrl_2g.yml \
  --no_submit=True
bash -n slurm_scripts/ms_reg_ctrl_2g.sh
```

Mechanics verified locally (generation only, scripts written outside the repo):

- All eight sweep configs resolve, share one key set, and each differs from `_ctrl_a` in exactly
  its declared axis (verified programmatically after writing them).
- The quoted `--override=` token survives `setup_job_cc.py`'s `split("=", 1)` kwarg parsing and is
  emitted as `--override compute_train_auxiliary_metrics=true seed=811923 wandb_name=...` — three
  values under **one** flag, which is what `nargs="*"` needs. Passing `--override` twice would keep
  only the last occurrence.
- `load_config` applies them (`runner.py:76-96`) with `yaml.safe_load`, so `true` becomes a bool.
- Staging for `model_type: mass_shatter` is config, template, `spec_df.pkl` (1.5 G), `mol_df.pkl`
  (22 M) and `data/split/nist20_ma_mi_inchikey` (8.7 M) — no `frag_dp`.
- The generated directives are `--mem=140G --cpus-per-task=6 --time=1:00:00
  --gres=gpu:h100_3g.40gb:1`, and the script passes `bash -n`.
- `.cu` sources ship via `pyproject.toml` package-data, and nothing in the training path imports
  `lmdb` or `sklearn` (neither is in the generator's pip list).

`w=online` is required for `j=` to work at all: the resume bookkeeping lives inside the W&B branch.

## 4. Read-out protocol

1. `E*` = the smallest final epoch across the eight runs. Compare **only** at `E*` and below.
2. Control reference = mean of `ms_reg_ctrl_a` and `ms_reg_ctrl_b` val cosine at `E*`. If those two
   differ by more than 0.007, stop (A3): re-run rather than interpret.
3. Per arm, at `E*`:
   - **Promote** (chain in Stage C): val cosine >= control + 0.007 **and** a non-worsening train/val
     cosine gap.
   - **Hold**: within +/-0.007. Explicitly *not* a null — see the bias below.
   - **Reject for this axis**: val cosine <= control − 0.007 **and** a train/val gap no better than
     the control's, i.e. it lost capacity without buying generalization.
4. Read the gap as the primary mechanism signal. `train_cos_sim − val_cos_sim` at matched epoch is
   the quantity every previous overfit claim in this family only proxied through the loss. Also log,
   per arm, the epoch of the val-loss minimum: on the unmerged side, val loss bottomed at epoch 15
   while val cosine climbed to epoch 82, so the two disagree about when to stop.
5. Train and validation log on **separate W&B steps** — `history()` returns NaN for `train_*` on
   validation rows. `groupby("epoch").last()` each side and join, or the gap is invisible.

### The screen is biased against the dropout arms

At ~40-55 epochs the runs are at 6-9% of the config's 630-epoch exposure-matched budget
(`min_epochs: 378`). Regularization characteristically trails early and wins late, so `_drop02` and
`_drop03` are the arms most likely to be wrongly rejected at one hour. That is the reason the gap,
not the cosine, is the primary read: the gap responds within the first tens of epochs. A dropout arm
that holds at `E*` **with a visibly narrower gap** is promoted to Stage C even without a cosine win.

The normalization arms carry the opposite property — `layer`/`graph` change the train/eval candidate
set from Jaccard 0.317 to 1.000 deterministically, from step one — so a one-hour box is a fair test
for them and the first trained-checkpoint measurement of that audit claim.

## 5. Risks and stop conditions

| Risk | Detection | Response |
| --- | --- | --- |
| OOM on the chosen slice (3g peak 26.9 GiB measured; 2g at `lb16` is a 15.8 GiB projection, not a measurement) | Stage A job dies | 3g: drop `loss_batch_size` to 32. 2g: drop it to 8 (14.1 GiB measured). Never mix profiles or `loss_batch_size` across arms within one sweep |
| 2g slice starves on 3 CPUs rather than on bandwidth, confounding the slice ratio | Stage A 2g pilot shows low GPU utilization with `num_workers: 3` | Read `system.gpu.0.gpu` on the 2g pilot before trusting the ratio; a CPU-bound 2g pilot understates the slice |
| The 2g pilot's slice ratio is confounded with its microbatch (256 vs 512), so it measures shape-vs-shape, not slice-vs-slice | By construction | Accept it: the shapes are what is being chosen between. Do not quote the ratio as a pure MIG benchmark |
| sm_90 native build fails or exceeds its 180 s timeout | Stage A error log | Fall back to the CPU candidate backend **for all arms**, and record it — a mixed-backend sweep is not a sweep |
| Instrumentation costs more throughput than expected, pushing `E*` below 20 | Stage A epochs/hour | Move the whole sweep to two-hour boxes before fanning out |
| Fewer than 20 epochs from staging overhead (1.5 G `spec_df.pkl` copy + ~4.3 G dataframe load) | Stage A timestamps | Same response |
| Reading an arm against the L40S history instead of the in-sweep controls | — | Compare only within this sweep; the GPU, the code and the instrumentation all changed |

## 6. What this sweep cannot answer

- **Whether merged val cosine transfers.** A merged arm scores against `tolerance_consensus`
  targets; and `merge_method: tolerance_consensus` (about 2026-09-03) redefined that
  target, so these numbers are not comparable to earlier merged runs, nor to unmerged runs, nor to FraGNNet.
  Promotion here means "worth a longer merged run", not "better model".
- **Convergent behavior.** Nothing at 6-9% of the schedule settles the 0.688-0.693 unmerged ceiling
  question.
- **Interactions.** Every arm is one knob. Dropout x normalization is a second sweep, gated on this
  one producing at least one promotion.
- **The BatchNorm residual-pinning mechanism.** `batch`, `layer` and `graph` all pin
  `||branch||/||residual||`; only pre-norm plus a zero-init gate would change that. If `_layernorm`
  wins, it is evidence about the candidate-set skew, not about branch attenuation.

## 7. Validation performed for this plan

- Full suite: **3,847 passed**, 1 skipped, 2 xfailed.
- `tests/test_spectrum_test_metrics.py`: 8 passed, including the two new range regressions. The
  out-of-range test fails with the production `AssertionError` when the fix is stashed.
- `tests/test_mass_shatter_model.py`: 104 passed, including the new
  `test_hydrogen_gain_is_bounded_by_the_complement` (intact molecule, and fragments with 1 and 5
  complement hydrogens). Two existing tests asserted the unbounded ladder and were updated, not
  deleted: `test_non_degenerate_cmf_uses_the_full_hydrogen_transfer_ladder` now exercises a
  component whose complement can actually donate the hydrogen, and the retained-channel test pins
  probability on the zero-shift rung instead of an unreachable `+1`.
- End-to-end forward passes emit no peak above the precursor.
- 202 passed across the metric, MASShatter-PL, merged-CE-context, explainability-demand and
  per-instrument-tolerance suites. Ruff on `spectrum_pl.py` reports the same 20 pre-existing
  findings before and after the change, none at the edit site; the test file is clean.

- All eight configs resolved through `load_config` against the current template; each diff is
  exactly the single key in the table above.
- Replicate floor, per-epoch times, epochs reached, peak GPU/host memory and GPU model pulled from
  the four historical W&B runs.
- Knob wiring confirmed by source line for all six arms; `cc_*` dropouts confirmed FraGNNet-only.
- LR-schedule forwarding confirmed still broken at `spectrum_pl.py:1766`.
- One job script generated locally and syntax-checked, outside the repository tree, to verify the
  override token, the resource directives and the staging list. **No script was generated in
  `slurm_scripts/`, and nothing was submitted.**

## 8. Next concrete check

Get the `spectrum_pl.py` fix onto the Fir checkout — the job stages code at start, so a queued job
would otherwise rerun the crash — then resubmit both Stage A pilots, and report the six Stage A
gates, above all the measured 2g-to-3g `min/epoch` ratio that selects the sweep shape. Do not fan
out until they pass.
