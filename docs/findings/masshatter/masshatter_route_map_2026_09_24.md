# Where MASShatter writes its per-molecule lookup: the route map

> **Status:** FINDING · **Updated:** 2026-09-24
> Inference-only, on two production checkpoints. Step 0 of the
> [memorization-route plan](../../plans/masshatter/masshatter_memorization_route_plan.md).
> The **global-attention message in the encoder** is the most memorization-specific route
> measured. Replacing each atom's molecule-specific global message with the batch average costs
> train 2.6 times what it costs val. The earlier probe's generalizing routes cost 1.2-1.3 times.
> The rewrite shrinks the eval-mode train/val gap from 0.18 to 0.04 while val falls only 0.08-0.09.
> The boundary block (1.46x) is where the key is read; global attention is where it is written.
> Three routes are small or null: the ion/H head (1.28-1.30x) is mostly generalizing, exact mass
> is weak (up to 1.44x, tiny damage), and `a_charge` precision is null. Both checkpoints agree
> to about 0.01 on every read.

## Scope and method

The checkpoints are `6h67zps9` (f1, cleaned target) and `3bej6fq7` (f0, raw target), the same
pair as the
[memorization probe](masshatter_memorization_route_probe_2026_09_23.md): unmerged `[M+H]+`/FT,
`data/split/nist20_inchikey`, seed 397623, best checkpoints from W&B in `saved_ckpts/`.

The probe is `scripts/debug/debug_pooled_embedding_memorization.py`, extended with new scopes.
Tests: `tests/test_debug_pooled_embedding_memorization.py`, 16 pass. All 17 conditions run on the
same batches, so every contrast is paired:

- **References:** `none`, and `cc_*` and `boundary_*` with `molmean` and `shuffle`, as before.
- **`gsa_*`, the molecule-level part of every atom embedding.** The global-attention message of
  both GINE_GSA encoder layers is replaced on a second encoder pass, through forward hooks on
  `gsa_norms`. `popmean` uses the batch mean for every atom, which removes identity. `molmean`
  uses the molecule's own mean, which keeps identity and removes per-atom detail. `xmol` uses
  another molecule's mean, which swaps identity. The rewritten embeddings replace the exact ones
  in the component pool and the boundary block only. The proposal reads the exact ones, so the
  candidate set and coverage do not move: val `wrecall` is 0.9651 (f1) and 0.9451 (f0) in every
  condition. The unit test checks that proposal logits and candidate m/z are bit-identical.
- **`ionh_*`.** The head's per-fragment ion/H logits are replaced by the molecule mean (every
  fragment of a molecule gets one H-shift pattern) or by the batch mean.
- **`feat_mass`, `feat_size` (cut count plus size) and `feat_charge`.** Each corrupts only its own
  descriptor columns.
- **`acharge_round`.** The continuous Gasteiger atom input, rounded to 0.1, through the second
  encoder pass.

Each checkpoint was run in eval mode, with train auxiliary metrics forced on, on 60 batches of
512 per split: about 30.7k spectra each, which is 60 of 308 train batches and 60 of 102 val
batches. Each checkpoint took 31-33 min on the local RTX 2070 SUPER. The runner is
`output/memorization_route_step0_2026_09_24/run_probe.sh`, and `analyze_route_map.py` beside it
reads the results (local, Git-ignored).

Two reads are used:

- **Damage ratio**: train drop divided by val drop. A route both splits use alike scores near the
  routes the earlier probe found generalizing (the pools, 1.21-1.24).
- **Excess gap closed**: how much more gap a condition closes than the pools do at the same val
  level, interpolating linearly through `none`, `cc_molmean` and `cc_shuffle`. This is the
  earlier probe's matched-damage read.

## Result

Cosine is `cos_sim_0.01`.

| condition | f1 train | f1 val | f1 gap | f0 train | f0 val | f0 gap | damage ratio (f1; f0) | excess gap closed (f1; f0) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `none` | 0.929 | 0.749 | 0.180 | 0.924 | 0.744 | 0.181 | — | — |
| `gsa_popmean` | 0.712 | 0.667 | 0.045 | 0.693 | 0.654 | 0.039 | 2.65; 2.58 | 0.118; 0.123 |
| `gsa_xmol` | 0.656 | 0.631 | 0.026 | 0.638 | 0.616 | 0.021 | 2.30; 2.25 | 0.130; 0.133 |
| `gsa_molmean` | 0.829 | 0.709 | 0.120 | 0.807 | 0.696 | 0.111 | 2.50; 2.46 | 0.052; 0.060 |
| `boundary_molmean` | 0.412 | 0.395 | 0.017 | 0.408 | 0.391 | 0.017 | 1.46; 1.47 | 0.083; 0.085 |
| `boundary_shuffle` | 0.227 | 0.214 | 0.013 | 0.237 | 0.223 | 0.014 | 1.31; 1.32 | 0.075; 0.070 |
| `cc_molmean` | 0.555 | 0.440 | 0.116 | 0.531 | 0.418 | 0.113 | 1.21; 1.21 | reference |
| `cc_shuffle` | 0.451 | 0.363 | 0.088 | 0.431 | 0.348 | 0.084 | 1.24; 1.24 | reference |
| `ionh_molmean` | 0.511 | 0.422 | 0.089 | 0.528 | 0.433 | 0.094 | 1.28; 1.28 | 0.020; 0.021 |
| `ionh_popmean` | 0.528 | 0.438 | 0.090 | 0.533 | 0.443 | 0.090 | 1.29; 1.30 | 0.025; 0.028 |
| `feat_size_molmean` | 0.769 | 0.618 | 0.151 | 0.765 | 0.613 | 0.152 | 1.22; 1.22 | 0.001; 0.002 |
| `feat_size_shuffle` | 0.690 | 0.550 | 0.140 | 0.680 | 0.543 | 0.137 | 1.20; 1.22 | −0.002; 0.002 |
| `feat_mass_molmean` | 0.906 | 0.733 | 0.173 | 0.908 | 0.732 | 0.176 | 1.44; 1.40 | 0.004; 0.002 |
| `feat_mass_shuffle` | 0.875 | 0.708 | 0.166 | 0.879 | 0.709 | 0.170 | 1.33; 1.29 | 0.005; 0.003 |
| `feat_charge_molmean` | 0.917 | 0.739 | 0.177 | 0.916 | 0.737 | 0.179 | 1.26; 1.31 | 0.000; 0.001 |
| `feat_charge_shuffle` | 0.908 | 0.732 | 0.176 | 0.909 | 0.731 | 0.178 | 1.23; 1.21 | 0.000; 0.000 |
| `acharge_round` | 0.928 | 0.749 | 0.180 | 0.924 | 0.743 | 0.180 | 1.30; 1.20 | 0.000; −0.000 |

The baseline reproduces the earlier probe (f1 train 0.928, val 0.739, on 100 batches) within its
sampling. `acharge_round` moves either split by at most 0.0005, so its damage ratio is noise.

## Reading

- **Global attention writes the key.** Take away the molecule-specific global message and train
  loses 0.22-0.23 while val loses 0.08-0.09. The gap falls from 0.18 to 0.04. At that val damage
  the pools would leave a gap of about 0.16. The boundary block needs a val drop of 0.35 to reach
  a similar gap.
- **The key is more than a molecule ID.** `gsa_molmean` keeps each molecule's own average message
  and removes only the per-atom variation. It still costs train 2.5 times what it costs val. What
  is stored is per-(molecule, atom) global context, not one identity vector.
- **The wrong identity hurts train most.** `gsa_xmol` has the largest train drop and excess, the
  signature of a lookup keyed on identity.
- **The global message contributes little to val.** Substituted by the population mean, val falls
  only 0.08-0.09. Global attention appears to serve mostly memorization. This is an inference-time
  substitution: a model trained without global attention could recover some of that val or keep
  some identity through its 2-hop local embeddings. Arm `r2` tests that directly.
- **The boundary block reads the key.** Its damage ratio (1.46-1.47) and excess (0.07-0.09) are the
  next largest. It is the sharpest per-candidate reader of the identity global attention writes.
- **The ion/H head is not a separate store.** Removing the per-fragment H-shift pattern costs both
  splits heavily, at a ratio of 1.28-1.30, close to the pools, with an excess of 0.020-0.028.
- **The descriptors are generalizing, with exact mass slightly asymmetric.** Cut count plus size
  is used heavily and symmetrically. `c_mass` has a ratio of 1.29-1.44 but moves val by at most
  0.04. `c_charge` is symmetric. The continuous precision of `a_charge` is not used as a key.

## Limits

- **Inference-time substitution, not retraining.** The substitutes are on-manifold in scale (means
  of real messages), but the network never saw them in training. Both splits pay that cost. The
  asymmetry is the read, and the conditions are compared with each other.
- **Two checkpoints, one seed, one task.** Unmerged `[M+H]+`/FT only. The merged lineage is
  unmeasured.
- **Sample.** 60 batches per split, not the full split. The paired design makes the contrasts
  much tighter than the absolute levels.
- **Both encoder layers are rewritten together.** Which layer's global message carries the key is
  not separated.

## What this changes in the plan

- **`r2_no_gsa` becomes the arm with the strongest evidence.** Read `wrecall` first, since the
  proposal reads the same encoder.
- **A targeted regularizer is now available.** Global-message dropout would replace the global
  message with the batch mean for a random share of molecules in training. It keeps global
  attention but stops the allocator relying on molecule-specific global context. It is not built.
- **Lower priority:** tying the ion/H head (plan Step 2e) and coarsening mass (2d).
  `a_charge` needs no discretizing.
- **`k1_key_dropout` is weaker.** It regularizes the head's input layers, but the key is written
  upstream in the encoder, and more allocator dropout (`h1`) was rejected at −0.0067
  ([ranking-transfer tail read](../../plans/masshatter/masshatter_ranking_transfer_plan.md#tail-read-from-wb-2026-09-24)).
