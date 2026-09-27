# Current project state

> **Status:** REFERENCE · **Updated:** 2026-09-27

## Current handoff

**Allocator-route probe on `c0`, `r2`, `nl1`, `nl2` read, and two masks on the allocator's set
built, `sm1` (message) and `sd1` (keys) (2026-09-27); committed, submitted 19:19 UTC on 09-27
(`7kz04hra`, `uxie2eko`), running.**

- **Probe** (inference only, 60 batches per split, eval mode):
  - The allocator's set message is the most train-specific route on `c0` (4.2x, excess gap
    closed 0.050), as on f0.
  - With global attention removed (`r2`) it carries almost twice the train-specific fit (train
    drop 0.132 against 0.070) while val barely uses it (0.027). This is where the lookup goes
    when its encoder route is cut.
  - `nl1` still writes the key in global attention (2.89x). In `nl1`/`nl2`, val leans on the
    set message too.
- **`sm1`** = `c0` + `masster_set_message_substitution: 0.3`. In training, each candidate, per
  allocator layer, with probability 0.3, receives its spectrum's detached mean set message
  instead of its own.
  - No parameters, bit-identical at eval.
- **`sd1`** = `c0` + `masster_set_key_dropout: 0.3`. In training, each candidate, per allocator
  layer, with probability 0.3, is hidden from the keys the inducing tokens summarize its set
  from.
  - The hidden candidate is still predicted and scored, so outputs and loss are `c0`'s. Dropping
    candidates from the output instead would overshoot shared-m/z isomers at eval.
  - Its prior is weaker than `sm1`'s (eval-time key hiding closed 0.013 of excess gap, against
    0.050), so the pair is read against each other: message or membership.
- **Checks.**
  - 17 test cases for the two options pass, and so do the related files (708).
  - The full suite passes on CPU with both options in: 4,450 passed, 107 skipped (CUDA-only), 2
    xfailed.
  - Both configs resolve to `c0` + their key, with 9,212,861 parameters.
  - Two-epoch CPU smokes exit 0.
- **Next:** at epoch 63, read `wrecall`, then the 54-63 tail against `c0` (+0.006
  single-seed rule), then `sm1` against `sd1`, then the probe on their checkpoints.

[Probe read](findings/masster/masster_memorization_energy_vs_parts_2026_09_26.md#set-message-probe-on-c0-r2-nl1-and-nl2-2026-09-27),
[plan: `sm1`](plans/masster/masster_memorization_route_plan.md#set-message-substitution-arm-sm1-built-2026-09-27-not-submitted),
[plan: `sd1`](plans/masster/masster_memorization_route_plan.md#set-dropout-on-the-allocators-keys-arm-sd1-built-2026-09-27-not-submitted).

**Encoder atom-mask and bond-dropout arms `am1`/`de1` built (2026-09-27); committed in
`9f596fc9`, not submitted.** These follow the user's proposal of atom drop and bond drop in the GNN. Each is `c0` +
one training-only key at 0.1 on the shared `GINE_GSA` encoder:

- **`am1`** (`masster_encoder_atom_mask`): an atom's projected input is replaced by a learned mask
  vector. It adds 384 parameters.
- **`de1`** (`masster_encoder_bond_dropout`): a bond leaves local message passing, both directions
  and every layer, and kept messages are scaled by 1/(1 − p). It adds no parameters.

This follows the audit's pattern: noise with every route left in place, as `k1`/`gm1`. It reopens
DropEdge, which the plan had rejected on the route-map argument the audit weakened.

- **Virtual node: not built.** `GINE_GSA` attention already covers it, and `r2` was null.
- **Validated.** 18 new test cases pass, and so does the full suite (4,535). Both configs resolve
  to `c0` + their key. Two-epoch local smokes exit 0.
- **Local screen (2026-09-27).** Four runs: `c0`, `am1`, `de1` and `c0` at seed 811237. They use a
  quarter of the molecules, batch 128 and 24 epochs, so they are not comparable to Fir.
  - `am1` is +0.0067 over the control mean on the val tail. It has the narrowest epoch-matched
    train/val gap: −0.0125, twice the one-pair control spread. That is a narrower gap with val up.
  - `de1` is −0.0053 on the val tail, and its gap is unchanged.
  - Caveat: one seed per arm.
  - A stronger mask does not help. At 0.3, val is −0.0053 against the control: it underfits,
    with the lowest train fit of any run. The rate ladder was stopped there (0.5 and 0.2 not
    read).
- **Allocator size and attention, local runs.** These follow the user's rule to start the set
  transformer small.
  - 16 inducing points and 4 heads: +0.0043, with train fit unchanged, so no underfitting.
  - Standard `full` attention with 4 heads (`fa_h4`): +0.0086, the best local run, with 1.1M
    fewer parameters.
  - New key `masster_intensity_mz_positional_encoding`, sinusoidal on neutral mass. `fa_h4_mz`
    was stopped at epoch 18 when the work moved to Fir. It was +0.0058 over `fa_h4` on epochs
    12-17, with train loss lower too.
- **Next:** on Fir run `submit_masster_mh_ft_memorization_route.sh am1`, plus `de1`
  optionally. Read `wrecall` first, then the 54-63 tail against `c0` (+0.006 single-seed rule).

[Plan section](plans/masster/masster_memorization_route_plan.md#encoder-atom-mask-and-bond-dropout-arms-am1-and-de1-built-2026-09-27-not-submitted).

**Memorization attempts audit (2026-09-27); read-only, no training.** Every attempt to close
MASSter's train/val gap, on both lineages, re-derived from code, the configs as run and W&B.
Nothing tested closes the ~0.18 eval-mode gap.

- **Real levers:** EMA (both seeds, both lineages), `k1` key dropout (both seeds) and merged LR
  decay. Removing a route has never helped.
- **Stale records, corrected in the finding and in place in the docs it names:**
  - `nl1` finished at +0.0037 (unresolved single-seed pass), not null.
  - `nl2` is −0.0021 against its same-code control `r2`, with train and val loss worse.
  - `sp0`/`sp1` reused their job names, so they resumed from the pre-fix epoch-2 checkpoint
    under the same W&B ids. They sit about 0.02 below `c0` at epochs 20-37.
  - `m2` is +0.0035 over `c0` at epochs 31-40, and +0.024 over `m1`.
  - `n2`/`n3` finished at −0.010/−0.009; `n4` runs at +0.0060 (epochs 51-60).
  - Merged arms recorded as unrun ran: `e6`, `c4a_ring_topology` and
    `c4a_complement_composition` are null. Two FraGNNet feature arms ran and are unread.
- **Method:** the +0.003 gate is about one standard error. The debug probe's val sample is a
  `spec_id`-ordered prefix that reads ~0.011 high. `r0` and the other structural arms change
  initialization.
- **Next:** submit `s1`; second seeds of `gm1` and `nl1`; resubmit `sp0`/`sp1` under new job
  names; gates 2-4 on `k1`/`gm1`/`u2`; read `m2` and `n4` at epoch 63.

[Finding](findings/masster/masster_memorization_attempts_audit_2026_09_27.md).

**Shared-proposal pathway audit (2026-09-26, 04:30 UTC on 09-27): the CE-context sampler was
broken, and `sp0`/`sp1` collapsed and crashed. Both bugs found are fixed, committed in
`1de12eb6`.**

- Every epoch ended with 28 one-spectrum batches of one molecule (mol 32341). 95% of the
  precursor-encoder BatchNorm running statistics then came from it. On `c0`'s checkpoint that
  alone drops val cosine from 0.7236 to 0.4771; a random-batch control gives 0.7233.
- `sp0`/`sp1` reached val cosine 0.03-0.24 (`c0` 0.58-0.66), with val loss 3-4 nats above train,
  and are marked crashed at epoch 2 with no log.
  - *Update 2026-09-27:* they were then resumed, not resubmitted. The resubmission reused the job
    names, so both resumed from their pre-fix epoch-2 checkpoints under the same W&B ids.
  - From epoch 3 they run the fixed sampler (312 steps per epoch). See the audit entry above.
  - **Final read, epochs 54-63 (2026-09-27):**
    - The sampler is null: `sp0` 0.7264 against `c0` 0.7276.
    - Sharing fails: `sp1` 0.7181, −0.0083 against `sp0`. Val candidate wrecall is 0.9205 against
      0.9383, and the proposal loss plateaus at 2.63 (train) against 1.87.
    - Deep-cut masks fall from 41% to 25%. Likely cause: one CE-blind product of Bernoullis cannot
      represent the energy mixture.
    - `sp1` is 22% faster per epoch but behind `sp0` at equal wall-clock after 1.5 h.
    - Next: per-NCE coverage from the Fir checkpoints before any remedy arm.
    - [§4](findings/masster/masster_shared_proposal_pathway_audit_2026_09_26.md#4-final-read-of-the-resumed-arms-sampler-null-sharing-fails-on-coverage-2026-09-27).
- **Sampler fix** (`data_sampler.py`, shared by the MASSter and FraGNNet CE-context samplers):
  - Molecules' chunks are merged in random order, and a repeated molecule's chunk waits for the
    next batch instead of closing this one.
  - On the `sp0` split: 312 batches per epoch instead of 376, and none under 64 spectra (58
    before). Each spectrum is still emitted once, with one chunk per molecule per batch.
- **Ion-mode fix** (`masster_model.py`): under sharing, `ion_mode_proposal_head` read the
  CE-conditioned context. It now reads the shared CE-free context. Dormant; no config uses it.
- One regression test each; the full suite passes (4,495).
- **Next:**
  - `sp0`/`sp1` finished as resumed runs (final read above). A fresh `sp0` is low priority, since
    the sampler null already survived the poisoned warm start. Any new CE-context arm needs a new
    job name, or it resumes an old `last.ckpt`.
  - Before Stage A (`c1`/`p1`/`p2`), re-measure its steps per epoch; its batches are fuller now.
- The sharing code itself is correct. Its merged-target loss equals the plain per-spectrum loss
  exactly.

[Finding](findings/masster/masster_shared_proposal_pathway_audit_2026_09_26.md).

**Sink-dropout arm `m2` built (2026-09-26); submitted 2026-09-27 (`41mnwm9f`), running.** At
epochs 31-40 it is +0.0035 over `c0` and +0.0238 over `m1`, so it tracks `c0`. `m1`'s memorization
sink applied to
a random half of the training spectra: new key `masster_memorization_sink_dropout`, default 0
(= `m1`), drawn per spectrum, no new parameters. `m1` failed at −0.0203 because the sink took
transferable fit too. With MemSinks-style mixing, the shared path must fit half of each
molecule's exposures alone. Read against `c0` and `m1`; it must recover 0.023 over `m1`, so it
is a long shot. There are 4 new tests, 323 pass in the related files, and both smokes exit 0.
Generate with `submit_masster_mh_ft_memorization_route.sh m2` after committing.
[Plan section](plans/masster/masster_memorization_route_plan.md#sink-dropout-arm-m2-built-2026-09-26-not-submitted).

**W&B read of the memorization-route and CE arms (2026-09-26, 03:45 UTC on 09-27); read-only.**
Running arms are projected to the epoch 54-63 tail by a fit over 21 finished arms of this recipe
(r = 0.96-0.98 from epoch ~20 on). Seed spread 0.0042.

- **`w1` (weight decay 0.1) is null:** +0.0010 over `c0`. Train fit drops by half as much as
  `k1`'s, and val loss moves inside the seed pair. Weight decay is closed on this lineage.
- **`u3` (switch EMA) fails:** −0.0089 against `u2`, below it at all 64 epochs, with train loss
  0.16 higher throughout. EMA works as an average kept off the training weights.
- **`n1` (`qet`) fails:** −0.0137 at epochs 24-33, projected −0.012; final tail −0.0121.
  - Its train loss equals `c0`'s, and only val loss is worse (+0.055). So the smooth CE input
    costs generalization, not fit, as the two-ladder reading predicts.
  - `n2`/`n3` are at epoch 9, too early to read. By loss, `n3` looks like `n1` and `n2` like `c0`.
    *Final 2026-09-27:* `n2` −0.0102 and `n3` −0.0085; both fail like `n1`.
  - **Conflict:** Enveda `i0` still uses `qet`; do not submit it on `qet`. Spectraverse cv0-cv9
    moved to `embed` + `int_embedder: fourier` on 2026-09-27 (see the Spectraverse entry below).
  - **Arm `n4`** (*since submitted as `shjgv9im`. At epochs 51-60 it runs +0.0060 over `c0`,
    against the null expected; read the tail at 63*): `c0` + `int_embedder: fourier`. It keeps the
    per-NCE code (`embed`) and swaps abs-sines for a sin + cos table with no aliasing in 0-300. No
    gain is expected on NIST20, where 76/180 is the one colliding pair and cost nothing; it prices
    the table swap. Resolves to that key alone; a 2-epoch smoke exits 0. To submit, commit the
    config and the submitter change, then on Fir run
    `submit_masster_mh_ft_memorization_route.sh n4` and `sbatch` the printed script.
- **`nl1`/`nl2` (light-side token) project null** (+0.0015, +0.0002) with train fit unchanged:
  removing the heavier-side pool does not dent the lookup, which moves to another route.
  - *Superseded by the final read (2026-09-27):* `nl1` +0.0037 is an unresolved single-seed
    pass. `nl2` is −0.0021 against `r2`, with train and val loss worse.
  - Neither arm can show rerouting, since both keep the boundary block, `c_mass` and the
    candidate set
    ([plan](plans/masster/masster_memorization_route_plan.md#final-read-of-nl1-and-nl2-2026-09-27)).
  **`nl3` (+ `k1`'s dropout) underfits at −0.017:** train loss +0.14, val loss up. `k1` needs
  redundant routes, so it does not stack with route removal. `nl3` later crashed at epoch 30;
  do not resubmit.
- `sp0`/`sp1` collapsed and crashed at epoch 2 (sampler bug; see the top entry), then resumed
  from their epoch-2 checkpoints after the fix.

Next: read `n2`/`n3` near epoch 30, and run the per-NCE-by-ladder read on `n1`'s checkpoint. Read
`nl1`/`nl2` at 63 and run the route map on them.
[Batch 3 read](plans/masster/masster_memorization_route_plan.md#batch-3-read-from-wb-2026-09-26),
[`n1`-`n3`](findings/masster/masster_qet_projection_candidate_union_2026_09_26.md#validation-arm-n1),
[`u3`](plans/masster/masster_ranking_transfer_plan.md#switch-ema-arm-u3-built-2026-09-25-not-submitted).

**ChemBERTa-3 basic model built (2026-09-27; committed in `d51fe50c`); not trained.** `model_type: chemlm`
fine-tunes ChemBERTa-3's first 6 of 12 layers (`DeepChem/ChemBERTa-100M-MLM`) end to end, with
the NEIMS binned head, on `ms_gym_benchmark_1_5`.
- **Runs.** E1 and T2 (random SMILES), one seed each, the user's call (2026-09-27).
  - E4 (NEIMS) is configured but not run, so E1 has no same-head baseline.
  - The configs are in `config/massspecgym/chemlm/`.
- **Environment.** `transformers` 4.46.3 and `tokenizers` 0.20.3 are installed in FRAGNNET-GPU,
  with the user's approval. The checkpoint is at `data/pretrained/ChemBERTa-100M-MLM`.
- **Validated.** The D8 gradient test and 17 other `chemlm` tests pass, and a CPU pipeline
  check ran T2 and E4.
- **NEIMS fix.** `NeimsModel` now passes ACE; before, every NEIMS config with `ace: True`
  failed.
- **Local GPU smoke.** One E1 epoch trained and validated at 1.30 s/step (fp32 with gradient
  checkpointing on the RTX 2070 SUPER) and saved a 2.05 GB checkpoint that loads.
- **Fir support.** `setup_job_cc.py` stages the checkpoint directory and installs the pinned
  Hugging Face packages, for chemlm jobs only. Every pin is in the Fir wheelhouse. `tokenizers`
  is 0.20.0 on Fir and 0.20.3 locally; the two give identical token ids on all MSGYM SMILES.
- **Checkpoints.** The runs keep theirs on disk (`upload_checkpoints: false`, the user's
  call): about 8 GB over E1 and T2.
- **Runner fix.** The runner used to copy un-uploaded checkpoints through `/tmp` around
  `wandb.finish()`, which filled the workstation's `/tmp` and lost a `last.ckpt`. It now renames
  them within the run folder.
- **Next: run on Fir, the user's choice.** Copy `data/pretrained/ChemBERTa-100M-MLM` into the
  Fir checkout, run a one-epoch smoke job, then E1 and T2.
  [Plan](plans/chemberta3_msgym_plan.md#built-2026-09-27).

**Enveda standalone MASSter baseline `i0` built (2026-09-26); not submitted, uncommitted.** The
user's call is to train Enveda alone for now.
`config/masster/enveda/enveda_180_inchikey_811_unmerged_mh_mmh_dual_ce_qet_ema999.yml` forks the
MSGYM EMA arm with these deltas:

- `[M+H]+`/`[M-H]-` through `filter_prec_types` (620,120 train spectra);
- `ce_insert_type: qet` with Enveda CE statistics;
- a 10 ppm QTOF loss tolerance with a 200 m/z floor;
- 20 epochs, 24,240 steps, about 4 h, unmeasured.

It resolves with only those deltas, and a local 1-epoch smoke on 2,000 spectra per split exits 0.
On this workstation, point `TMPDIR` off the full 3.7 GB `/tmp`, or the checkpoint save fails.

- **Isotope ceiling (Stage 0a step 2).** A perfect monoisotopic model reaches 1 − cos = 0.0072 on
  Enveda val: 0.0086 positive, 0.0025 negative, 0.022 on Cl/Br molecules.
- **Plan corrections.** The isotope plan's 5 ppm tolerance was too tight: the median spectrum's
  fragment error is 3.8 ppm, p90 6.7.
- **Next:** a second seed for the Enveda floor, and Stage 0a step 3.

[Plan](plans/masster/masster_isotope_envelope_plan.md#current-state-and-next-check).

**Light-side token arms `nl1`-`nl3` (2026-09-26); finished. Final read in the
[plan](plans/masster/masster_memorization_route_plan.md#final-read-of-nl1-and-nl2-2026-09-27).**
Each candidate is represented by the lighter side of its cut (the neutral loss of a large
fragment). The heavier side enters only as element counts (`c_heavy_side`), because the side-of-cut
probe found it the more memorizing input.
- `nl1`: `c0` + the token.
- `nl2`: `nl1` + `mol_gnn_type: GINE`, which leaves no whole-molecule representation in the token.
- `nl3`: `nl2` + `k1`'s dropout.

[Plan section](plans/masster/masster_memorization_route_plan.md#light-side-token-arms-nl1-nl3-built-2026-09-26-not-submitted).

**Memorization: energy vs parts (2026-09-26); inference only, on f0 `3bej6fq7`.**
The route-map probe gains a `ctx` scope (CE removed, or NCE ±1/±5, intensity head only) and a
per-spectrum dump. A per-peak analysis rates each break by how many other train molecules have it.

- **The key is per molecule, not per energy.** ±1 NCE costs train and val alike. Removing CE is
  mildly train-specific (excess 0.045, a third of global attention's), mostly above NCE 60.
- **Train fit is a whole-molecule lookup; val transfer runs through parts.** Within a spectrum,
  unseen-break peaks cost +0.026 in error on train and +0.097 on val, at fixed molecule
  similarity. 28% of val fragment-peak intensity is on unseen breaks, captured at 0.49.
- **NCE values of the two ladders are not interchangeable** (0.71-0.84 on the same molecule), so
  a smooth energy input needs a ladder input next to it.

[Finding](findings/masster/masster_memorization_energy_vs_parts_2026_09_26.md).

**QET breakdown-curve gates for MASSter (2026-09-26), and a fix for the CE input; `qet` committed
(`1494441e`), `qet_mlp` and arms `n2`/`n3` uncommitted.**
Offline analyses on `u2_ema999`, unmerged `[M+H]+`/FT, all val spectra plus 2,000 train
molecules (scripts in `output/qet_projection_2026_09_26/`).

- **Enforcing the law as an output constraint is worth about +0.002.** `u2`'s own predictions,
  projected per molecule onto the one- or two-component form, gain +0.0019 val (SE 0.0002) and
  lose 0.0009 on train. The model's CE series are already QET-consistent.
- **One CE-blind candidate set per molecule is feasible.** The 256 best of the union (median
  392 fragments) cover 0.9422 of target intensity against 0.9402 per energy, and 384 cover
  0.9471. That makes a once-per-molecule pass possible; its speedup is not measured.
  - For `share_proposal_across_ev`, ranking by the mean per-energy proposal probability (what
    its merged-target loss should learn) covers the same and loses no energy bin, including
    each molecule's lowest and highest NCE. CE balancing adds nothing.
  - Most of the union is same-m/z isomers: full coverage needs a median 26 fragments. Skipping
    fragments that add no new channel m/z, 128 candidates cover 0.9428, more than each energy's
    own 256. That is target-free, but its cosine value is unmeasured; a mask-order U128 already
    tied U256 on cosine.
  - **Arms `sp0` (`85mfkk8i`) and `sp1` (`bu6dzjph`) are void: the sampler's epoch tail
    poisoned BatchNorm, and both crashed at epoch 2. The sampler is fixed (`1de12eb6`).
    Resubmit both under new job names; reusing the names resumed these ids from their pre-fix
    checkpoints
    ([audit](findings/masster/masster_shared_proposal_pathway_audit_2026_09_26.md)).** `sp0` = `c0` + the CE-context batch
    sampler (whole molecules per batch: 32 spectra per context, edge budget never binding, 512
    spectra cap as in `c0`); `sp1` = `sp0` + `masster_share_proposal_across_ev`. Read `sp0`
    against `c0` and `sp1` against `sp0` on the epoch 54-63 tail, plus seconds per epoch.
    - Both 2-epoch local smoke runs on a 60-molecule whole-molecule split exit 0; `sp1`
      really grouped (60 ladders in 85 chunks).
- **The CE input was categorical in every model.** `ce_insert_type: embed`/`embed_clamped` with
  the default `abs-sines` table (749 configs, all MASSter and FraGNNet lineages) gives each
  integer a fixed binary-like code.
  - NCE x and 256 − x collide, so 76 and 180 get identical inputs; at `ace_max: 100`, 30 and
    98 eV do.
  - On `u2` an unseen NCE drops val cosine to 0.22-0.42, and rare high NCEs gain +0.03 to +0.07
    from projection.
- **Fix: new `ce_insert_type: qet`.**
  - It feeds the raw CE plus a missing-CE flag through a learned linear layer
    (`QETEnergyEncoder` in `base_model.py`). NCE goes on a standardized `1/(NCE + 200)`, the
    breakdown-curve axis. ACE (eV) is linear: on NIST20 QTOF eV series (1-50 eV) linear eV fit
    best, and `1/(NCE + 200)` on converted eV predicted held-out energies 0.0009 worse
    (SE 0.0001). It works in the shared `CEModel` and in `MassterContextEncoder`.
  - Existing configs and checkpoints are unchanged.
  - A probe on held-out NCE values chose it: 0.9845 against 0.9480 for abs-sines and 0.9627 for
    32 RBF bumps.
  - 18 new tests in `tests/test_ce_qet_embedding.py`; 287 pass with the related CE and model
    test files.
- **Arm `n1` (`c0` + `qet`, `jxczo2kf`) fails; see the top entry for the epoch 24-33 read.** Over epochs 10-13 of 64 it
  averaged 0.7009 against 0.7127 / 0.7137 for the `c0` seeds (−0.0124; seeds 0.001 apart).
  Candidate coverage equals `c0`'s, so the loss is intensity allocation.
  - Still to read on its checkpoint: `nce180_check.py` (unseen NCE 170/190, 76 vs 180) and
    per-NCE loss by acquisition ladder.
  - Hold "new configs should use `qet`" (Spectraverse, Enveda) until `n1`-`n3` are read.
- **`n2` (`zov75vk3`) and `n3` (`o9pz0j9j`) running** (epoch 9 at 03:45 UTC on 09-27).
  - `n2` = `n1` + new `ce_insert_type: qet_mlp`: the same input through
    Linear(2, 32) → SiLU → Linear(32, 32). `qet` checkpoints still load.
  - `n3` = `c0` + `ce_insert_type: id`: (NCE − 60)/40 straight into the context projector.
    `n1`'s affine embedding makes it this scalar on a curved axis, so `n3` ≈ `n1` would put the
    cost on losing the per-energy code.
  - Checked locally: 4 new tests plus an extra invalid-setting case; 199 pass across the QET,
    merged-CE, CE-location and MASSter model test files. Both configs resolve to their base plus
    `ce_insert_type` and naming, and 2-epoch smokes of both exit 0.
  - The template NCE scale (60, 40, max 200) matches the NIST20 `[M+H]+`/FT split (train mean
    62.2, std 42.6, 2-195).
  - A once-per-molecule head is a compute restructuring, not an accuracy fix.
- **Other integer embeddings checked.** No other table aliases in production.
  - FraGNNet's `base_formula` holds heavy elements only, with C at most 129.
  - `depth`, `rdbe`, cut bonds and the H-channel stay far below the aliased range (129 and up).
  - DAG `out_degree`/`in_degree` do alias hub nodes (0.5% / 0.2% of nodes), but only a debug
    config enables them.

[Finding](findings/masster/masster_qet_projection_candidate_union_2026_09_26.md).

**Enveda isotope probe (2026-09-25); read-only, uncommitted script.**
`scripts/debug/debug_enveda_isotope_probe.py` ran on 60,000 records of each export.

- 13C satellites hold 7.7% of positive-mode training-target intensity and 3.9% of negative-mode.
  Halogen M+2 partners hold another 6.2% on Cl/Br molecules. Every config predicts monoisotopic
  peaks only, so that intensity is unreachable.
- Co-isolation: M+1 passes at 0.93 of natural in positive mode and 0.48 in negative. M+2 is a
  per-molecule switch that structure cannot predict: it passes in 38% of positive-mode halogen
  spectra.
- This timsTOF separates 13C from an H shift below m/z 300, where the detector's QTOF gate assumes
  it cannot.
- The release's MS1 isotope fields are extracted at 13C spacing, so they miss the halogen M+2.
- MASSter cannot predict any of it: `num_isotopes` is FraGNNet-only. The follow-up is the
  [isotope envelope plan](plans/masster/masster_isotope_envelope_plan.md) (PROPOSED), whose next
  step is a model-free ceiling on Enveda and NIST20 val.

[Finding §9](findings/enveda_180_dataset_2026_09_25.md#9-isotopes-2026-09-25).

**`[M+Br]-` adduct supported (2026-09-25); uncommitted.** It is added to the four adduct tables
in `formula_utils.py` like `[M+Cl]-`: retention keeps Br, and CMF loses HBr to give `[F-H]-`.
Everything downstream reads those tables. Four tests were added or extended, and the full suite
passes (4,448). On Enveda's 25 `[M+Br]-` spectra, the precursor matches within 3.2 ppm, and
Br-retaining fragments carry most of the intensity. NIST23 has 224 such spectra; no existing split includes any, and the
Enveda split would need regenerating to pick up its 25.
[Finding §8](findings/enveda_180_dataset_2026_09_25.md#8-mbr--support-2026-09-25-uncommitted).

**Memorization-route batch 2 read, all ten runs (2026-09-26); weight-decay arm `w1` finished
null (top entry).** W&B tails over epochs 54-63.

- **The seed spread is 0.0042** (`c0_s2` − `c0`, seed only): above the +0.003 gate and twice the
  0.0022 used so far. A single-seed delta under about +0.005 is not resolved.
- `k1` and EMA replicate at seed 811237 (+0.0048, +0.0072). `k1_ema` has the best tail on this
  lineage (0.7421, +0.0145 over `c0`) but is +0.0029 over `u2`, so stacking is not shown. `gm1`
  +0.0051 on one seed; `gm2` −0.0029. Coverage moves by at most 0.0007.
- The swap arms `gm3`-`gm6` all fail (+0.0023, +0.0006, −0.0029, −0.0025), each below `gm1`:
  another molecule's real message does worse than the batch mean. The swap family is closed: a
  resource decision on one seed, with each swap delta inside the seed spread.
- Still open: gates 2-4 (ranking decomposition, route map, novelty bins) on the checkpoints
  that passed gate 1 (`k1`, `k1_ema`, `gm1`, `u2`).
- `w1_wd10.yml` is `c0` with `weight_decay` 0.1. AdamW's decay timescale falls from ~325 epochs
  to ~32, inside the 64-epoch run. Weight decay was never varied on this lineage; on merged,
  `wd10` was the one arm that narrowed the gap without lowering train. It resolves to that key
  alone, and a debug smoke confirms the optimizer group. Committed in `4e29da7c`.

[Batch 2 read](plans/masster/masster_memorization_route_plan.md#batch-2-tail-read-six-finished-runs-2026-09-26),
[`w1`](plans/masster/masster_memorization_route_plan.md#weight-decay-arm-w1-2026-09-25).

**MASSter Spectraverse CV configs built (2026-09-25, committed in `4e29da7c`). CE input went to
`qet` on 2026-09-26 (`66b18dd8`), then to `embed` + `fourier` on 2026-09-27 (uncommitted). Fir
submission prepared for cv0-cv3; not submitted.**
`config/spectraverse/masster/cv0.yml`..`cv9.yml` are unmerged and forked from the MassSpecGym
`_ema999` recipe. The folds differ only in `wandb_name`, `split_dp` and `lr_total_steps`
(40 × ceil(train / 512), 30,120-31,200). The cv0 header records the measurements behind each key.

- **Spectraverse deltas.** `masster_include_cmf: true`, because `[M+NH4]+` and formate/acetate
  spectra fragment by losing the adduct, which the ±4 H ladder cannot reach. All nine adducts;
  `inst_types` FT/QTOF/Other/NA, with FraGNNet's 20 ppm / 500 Da tolerance for Other and NA;
  40 epochs. The 44% of rows with no CE train on the CE table's learned missing-CE token.
  CMF is live for `[M+Na]+`, `[M+NH4]+`, `[M+K]+`, `[M+HCOOH-H]-`, `[M+CH3COOH-H]-` and
  `[M+Cl]-`. For `[M+H]+`/`[M-H]-`, CMF duplicates a CRF H shift, and `[M]+` has no CMF shift, so
  those three emit no CMF channel. Unmerged, because
  no merged MASSter run has trained CMF with dual CE, so the comparator is
  `config/spectraverse/fragnnet`, not `fragnnet_merged`.
- **CE input `embed` + `int_embedder: fourier`, 2026-09-27** (MassSpecGym: `embed` with
  abs-sines; 2026-09-26 on this config: `qet`). `qet` is dropped because `n1`-`n3` all fail on
  NIST20 (above) with train fit unchanged. `fourier` keeps the per-energy code, and `n4` is its
  NIST20 test. At max 300 it gives each integer 0-300 its own row, into `Linear(22, 32)` per CE
  channel, where abs-sines aliases 44 pairs in 212-300. It does not fix rare or unseen derived NCE
  values (525 distinct). Against the pre-`qet` configs, `int_embedder` is the only changed key.
- **`loss_batch_size: 4`, measured.** The recipe's 64 runs out of memory at batch 32 on the 8 GB
  card: each loss chunk pads to its widest target, and spectraverse peaks are p99 783, max 4096.
  At batch 128, `lb4` peaks at 4.68 GiB and 0.346 s/step, against 4.25 GiB and 0.291 s/step for
  the MassSpecGym recipe at `lb64`. Batch 512 was not run.
- **Validation.** All ten resolve through `load_config`. cv0's resolved diff against the
  MassSpecGym `_ema999` config is exactly the intended keys. The real cv0 train and val datasets
  construct (387,399 / 46,057 rows, none dropped, 757 steps/epoch). A 2-epoch smoke on a
  512-spectrum subsample exits 0 (val cosine 0.094 -> 0.167). With `qet` the same smoke exits 0
  (0.082 -> 0.169), and its checkpoint holds a `Linear(2, 32)` encoder on each CE channel. With
  `fourier` it exits 0 too (0.106 -> 0.162, 2026-09-27). Run
  local smokes with `TMPDIR` off `/tmp`: the post-fit checkpoint copy fails on the 3.7 GB volume.
- **Not measured.** The 40-epoch horizon (sized from the MassSpecGym and NIST20 peaks in steps),
  batch-512 memory, and wall clock (estimated ~6 h per fold).
- **Next:** commit the configs, then on Fir generate the cv0-cv3 array (`setup_job_cc.py
  --cv_config_dp config/spectraverse/masster --cv_start 0 --cv_end 3 --cv_use_job_id`, job name
  `msh_spv`, `fir:h100-3g.40gb`, 12 h; see the
  [submit guide](guides/submit_cluster_runs.md#4-cross-validation-and-existing-batch-launchers)).
  Read peak GPU memory and best epoch on cv0 before cv4-cv9.

**Learning-rate-schedule arms `s1` and `s1_k1_ema` built (2026-09-25); not submitted
(committed in `4e29da7c`).** Plan decision C, config only. Every MASSter run so far trains at a constant
`lr: 1e-3` with no warmup. Both arms add `d3`'s schedule: a 1000-step warmup, then cosine to 1% of
peak at step 19,712 (64 epochs × 308 steps).

- `s1_lr_cosine.yml` is `c0` plus the four schedule keys, read against `c0` (`vcpho5qm`) and `u2`.
- `s1_k1_ema_lr_cosine.yml` is `k1_ema` plus the same keys, read against `k1_ema` (`j4u25qfd`);
  optional, since `k1_ema` missed the stacking gate (read above).
- Each resolves to only those keys against its control. Two debug smokes exit 0; the one with EMA
  shows the scheduler stepping once per optimizer step, at the exact cosine rate.

Generate on Fir with `submit_masster_mh_ft_memorization_route.sh lr`.
[Plan](plans/masster/masster_memorization_route_plan.md#learning-rate-schedule-arms-s1-and-s1_k1_ema-2026-09-25).

**Enveda-180 preprocessed, 8:1:1 split `data/split/enveda_180_inchikey_811` (2026-09-25);
reader committed in `4e29da7c`.** The split is by InChIKey first block, with 143,568 / 17,946 / 17,946
molecules and 640,475 / 79,881 / 80,152 spectra. It keeps the six adducts the mass table
supports and drops dimers and the misnamed formate/acetate spectra
([§7](findings/enveda_180_dataset_2026_09_25.md#7-split-datasplitenveda_180_inchikey_811-2026-09-25)).
Background from step 2 follows. New
`--input_format enveda` in `preproc_scripts/01_prepare_df.py` reads the filtered JSONL export.
`data/df/enveda_180_df.csv` and `data/proc/enveda_180/` hold 1,158,293 timsTOF QTOF spectra of
183,476 molecules, with nothing dropped. The filtered export is exactly a 1% base-peak floor.
Multi-energy records are stepped acquisitions, and the 1,946 two-energy ones read as `ramped`
downstream, so split with `--ce_types stepped single`. Dimers are 28.7% of spectra, and the
molecules are effectively disjoint from NIST20/23, MSGYM and Spectraverse (26-66 shared). Step 2
peaked at 23.3 GB RSS plus swap on this workstation. Next: decide whether Enveda trains with
NIST/MSGYM or alone, then write the first config.
[Finding](findings/enveda_180_dataset_2026_09_25.md).

**MASSter exports per-peak fragment annotation (2026-09-25); uncommitted.** It uses the same
columns and flags as FraGNNet (`--return_formula_str`, `--return_cc_hash`): a neutral formula
plus adduct per peak, P(n|f), and an atom bitmask per fragment.

- **Model.** `MassterModel.forward(return_annotation=True)` adds per-pathway fragment, ion-mode
  and H-shift indices and the fragment atom memberships. `MassterPL` turns it on from the
  exporter's `return_explainability_tensors`, so training is unchanged.
- **Export.** `masster_annotation_tables` in `script_utils.py` groups the pathways into
  FraGNNet's formula / joint / node tables, and the existing filter, rebase and HDF5 path
  writes them. `aggregate_peaks_exact_mz` gained `return_inverse`.
- **Validation.**
  - `ll04i00m` on 128 MSGYM val spectra (CPU; the GPU was taken by another job): m/z
    reproduced to 5.7e-13 Da, and every fragment is a connected substructure with the
    formula's heavy atoms. The spectrum is identical to the annotation-off export.
  - On GPU (512 spectra), the annotation adds no measurable time. The HDF5 grows by 46 KB per
    spectrum at `--min_intensity 0` and by 1.8 KB at 10.0.
  - 15 new tests in `tests/test_inference_masster_annotation.py`.
- **Behavior change.** `scripts/run_inference.py` defaults both flags to True, so MASSter runs
  through it now build the annotation. `run_save_inference_eval.py` defaults to False.
- Conventions and costs: [run_inference guide](guides/run_inference.md#fragment-annotation).

**MASShatter renamed to MASSter (2026-09-25); uncommitted, not on any cluster yet.** 466 files
edited and 408 moved. Code, configs and docs now say `model_type: masster`, `masster_*` keys,
`MassterPL`/`MassterModel`, `fragnnet.frag.masster_*` and the `masster_core` Cython extension.
Directories: `config/masster/`, `docs/{findings,plans}/masster/`, `sweep/masster/`, and
`tests/test_masster_*`.

- **Old names still load.** `migrate_legacy_masster_name` in `src/fragnnet/utils/pl_utils.py`
  renames `mass_shatter` values and keys; the current key wins when both are present. It runs in
  `get_pl_hparams`, in `MassterPL.__init__` (so Lightning's `load_from_checkpoint` too), and in
  `load_config` (experiment YAMLs, templates saved in checkpoints, CLI overrides), plus on W&B
  resume and sweep configs. Two debug scripts that read checkpoints or W&B runs directly
  (`analyze_proposal_fragment_coverage.py`, `probe_masster_regularization.py`) now migrate
  first. [Architecture](architecture.md#entry-points-and-data-flow).
- **Kept the old spelling.** These name existing runs and artifacts: W&B `wandb_group` values
  (`nist20_mass_shatter_l40s` and 23 others), past `wandb_name` and sweep names, `output/`
  paths, and branch names. New runs log `model_type: masster`, so a W&B query spanning both
  eras needs both values and both key spellings, as `probe_masster_regularization.py` does.
- **Validated.**
  - Full suite: 4,394 passed (the 4,390 baseline plus 4 alias tests). Ruff is clean on the 82
    changed Python files.
  - Checkpoints: `3bej6fq7`, `pkavjg8z` and `5qws8g9x` load strict, with every old key's value
    and identical weights.
  - Configs: 255 of 264 MASSter configs and all 10 sweeps resolve. The 9 failures are in
    `config/masster/legacy/`; they set `exploration_*` keys the template had already dropped.
  - Docs: no doc link broke (107 broken before and after).
- **Rebuild and cluster.** Local checkouts need `python setup.py build_ext --inplace`. The old
  `mass_shatter_core*.so` stays for branches that still import it. Generated Fir/Vulcan jobs
  build the package themselves. Do not pull the rename into a cluster checkout while queued jobs
  still name `config/massshatter/…`.
- **Sessions started before the rename hold the old paths.** One wrote the batch-2 configs to
  `config/massshatter/` minutes before it; they are now in `config/masster/`. The ignored
  `.claude/settings.json` still allowlists the old test paths.

**Switch-EMA arm `u3` (2026-09-25); ran and fails (top entry).** `u2`'s EMA only changes
which weights are validated, so it cannot touch the lookup. Switch EMA copies the average into
the training weights at the start of every epoch. It is a new template key,
`ema_switch_every_n_epochs` (default null, requires `ema_decay`), in `ExponentialMovingAverage`.
The arm is `u3_switch_ema999.yml`, `u2` plus that key at 1, read against `u2` (`c8hh9j1b`, 0.7392).
Risk: at 0.999, a per-epoch switch keeps ~14% of each late epoch's steady movement, so it may
underfit; the fallback is a 4-epoch interval. 6 new callback tests (resume bit-identical to an
uninterrupted run) and a CPU smoke pass. Generate on Fir with
`submit_masster_mh_ft_ranking_transfer.sh u3`.
[Plan](plans/masster/masster_ranking_transfer_plan.md#switch-ema-arm-u3-built-2026-09-25-not-submitted).

**Global-message swap arms `gm3`-`gm6` built (2026-09-25); all fail, read in the top entry.** A
feature-level CutMix that keeps the label: in training, the encoder's normalized global-attention
message is replaced by another molecule's real, detached message, where `gm1` uses the batch
mean. The head cannot tell when a swapped key is wrong, which it can with the mean. Labels are
not mixed: a MASSter target is an allocation over one molecule's candidates.

- New keys `masster_global_message_swap` (probability) and `masster_global_message_swap_unit`
  (`atom`, `molecule` or `feature`), in `GINE_GSA._swap_global_message`. Default-off, no
  parameters, nothing drawn at 0, so `c0` (`vcpho5qm`) is the control.
- Arms: `gm3` atom 0.3 (read against `gm1` too), `gm4` molecule 0.3, `gm5` feature 0.3, `gm6`
  molecule 0.6. Each resolves to only its swap keys against `c0` (9,212,861 parameters), and the
  2-epoch smokes exit 0.
- Four arms against a 0.0022 spread: replicate any pass. `r2`'s null means the key may re-form
  locally; gate 3 (route map on the checkpoint) checks it.
- Generate with `submit_masster_mh_ft_memorization_route.sh swap` on Fir; `batch2` now includes
  them. [Plan](plans/masster/masster_memorization_route_plan.md#global-message-swap-arms-gm3-gm6-built-2026-09-25-not-submitted).

**Batch 2: all ten finished 2026-09-26 (read in the top entry).** At ~21:45 UTC on 2026-09-25 the original six were
at epochs 49-53 of 64 and `gm3`-`gm6` at 9-13. Early read at epochs 23-27 (20:30 UTC), not the
gate: `gm1` +0.0064 over
`c0` but narrowing, `gm2` −0.0030 and widening, `k1_ema` +0.0039 over `u2`, and both replicates
repeat their originals (`k1_s2` +0.0051, `u2_s2` +0.0122 against `c0_s2`, which sits +0.0011 from
`c0`). Run ids and the table are in the
[plan](plans/masster/masster_memorization_route_plan.md#batch-2-configs-2026-09-25-not-submitted).
**Eval-mode train/val gap on the Step 1 checkpoints:** about 0.18 cosine everywhere (`c0`@63:
0.922 train, 0.739 val). `k1` closes 0.017 of it, EMA (`u2`) lifts train and val together, and `r2`
leaves it in place (−0.007). Coverage differs by ≤0.004, so the gap is allocation
([table](plans/masster/masster_memorization_route_plan.md#eval-mode-trainval-gap-on-the-step-1-checkpoints-2026-09-25)).
Configured as ten jobs:

- `gm1` and `gm2` (below), and `gm3`-`gm6` (above);
- `k1_ema` (`k1` + EMA), read against `u2`;
- `c0_s2`, `k1_s2` and `u2_s2` at seed 811237, replicating `k1` and EMA against a matched control.

The three replicates do not upload checkpoints. Each config resolves to exactly its keys, and a
`k1` + EMA smoke exits 0. Generate on Fir with
`submit_masster_mh_ft_memorization_route.sh batch2`.
[Plan](plans/masster/masster_memorization_route_plan.md#batch-2-configs-2026-09-25-not-submitted).

**Global-message noise arms `gm1`/`gm2` built (2026-09-25); running in batch 2.**
Training-only noise on the encoder's global-attention message, where the route map found the
key is written. The code is in `GINE_GSA`, after the message's norm, and reaches the shared
precursor encoder only.

- `gm1`: `masster_global_message_substitution: 0.3` replaces an atom's message with the
  detached batch mean.
- `gm2`: `masster_global_message_noise_std: 0.5` adds Gaussian noise scaled to each
  feature's batch standard deviation.

Both keys are default-off, draw nothing at their defaults and add no parameters, so `c0`
(`vcpho5qm`) is the control. Substitution is drawn per atom; the plan's per-molecule form is not
built. There are 14 new tests, and the full suite passes (4,390). Each config resolves to one key
against `c0` (9,212,861 parameters), and the 2-epoch smokes exit 0. Generate with
`submit_masster_mh_ft_memorization_route.sh gm` on Fir.
[Plan](plans/masster/masster_memorization_route_plan.md#global-message-noise-arms-gm1-and-gm2-built-2026-09-25-not-submitted).

**Memorization-route Step 1 read from W&B (2026-09-24); read-only.** All six runs finished
64 epochs in group `nist20_mh_ft_memorization_route_2026_09_24`, with checkpoints as run files.
Tail (54-63) `val_cos_sim_0.01` against `c0` (`vcpho5qm`, 0.7276):

- `k1_key_dropout` (`wg0nxymi`): **+0.0074**, above `c0` at 57 of 64 epochs; `wrecall` held.
- `r2_no_gsa` (`9lt5s2u8`): +0.0006, a null with 1.18M fewer parameters.
- `r1_local_cut`: −0.0029. `r0_no_boundary`: −0.0129. `m1_sink`: −0.0203, since the sink absorbs
  transferable signal.

`c0` sits 0.0022 below `f0` (`3bej6fq7`) on a config differing only in default-off keys, so
that is the replicate floor. `k1` passes gates 1 and 5. Gates 2-4 need the route map and novelty
bins on its checkpoint, and train-mode losses cannot rule out underfitting. Next: the route map
on `k1`/`r2`/`r0`, a `k1` seed replicate, and a `k1` + EMA arm.
[Plan read](plans/masster/masster_memorization_route_plan.md#step-1-tail-read-from-wb-2026-09-24).

**MASSter exports now hold one row per peak (2026-09-24); uncommitted, behaviour change.**
`process_batch_to_tabular_row` (`src/fragnnet/utils/script_utils.py`), which every exporter
calls (`run_save_inference_eval.py`, `run_inference.py`, `run_save_inference.py`), used to write
MASSter's candidate rows: about 4.5 connected fragments share each peak's m/z at the 0.1% cut.
For `MassterPL` only, it now sums candidates at one m/z with `round_aggregate_peaks` (4
decimals, `sum_ints` aggregation), the step every score already takes, before the base-peak
rescale and before `min_peak_int`. So exported peak counts are peak counts, the filter acts on
summed peaks (a peak whose candidates are each below it but sum above it is kept), and the base
peak is 1000 over peaks. Scores are unchanged. Peaks are sorted by m/z per spectrum (exact m/z
kept; see the update below). On `3bej6fq7` test at the default `--min_peak_int 1.0` the median is 22-24
peaks per spectrum against 20 true, where candidate rows gave 105. A 20-batch real export matches
the offline peak-level 0.1% export on every spectrum whose prediction did not change between
runs (up to float32 summation, ≤ 0.22 on the 0-1000 scale). Exports made before this carry
candidate rows. Tests: 6 new in `tests/test_inference_masster_peak_aggregation.py`;
`test_masster_cli_exports_without_dags` now expects summed peaks; full suite 4,373 passed.
**Update (2026-09-24, late):** the export now calls `aggregate_peaks_exact_mz` (same 4-decimal
groups, exact m/z kept, the intensity-weighted mean where members differ), not `round_aggregate_peaks`. The
rounded m/z did change scores: on MSGYM test (`ll04i00m`), binned `cos_sim_0.01` moved on 378
spectra (731 for sqrt; means +0.0001) as peaks crossed 0.01 Da bin edges, while the Hungarian
scores matched. With exact m/z, an unfiltered aggregated export reproduces all six in-model
scores to 4.8e-7 (median 924 peaks per spectrum against about 2,060 candidate rows).

**Matched intensity filter and target cleanups measured on `3bej6fq7` test (2026-09-24); offline,
uncommitted.** One unfiltered test prediction; every variant is a filter on the target. Dropping
peaks at or below 0.5% / 1% of the base peak from both prediction and NIST20 target costs −0.0003 /
−0.0009 `cos_hun`, the same on every target. The f1 cleanup adds +0.0050, mostly at high NCE. The
literal ALEA-style cleanup (20 ppm centroid, fixed 5 ppm subformula) lowers the score by −0.025:
without an m/z floor it deletes real low-mass fragments. The `output/cross_eval_mh_ft_clean_2026_09_23/`
directory (exports, targets) was removed from the workstation this evening; the new test prediction
and targets live in `output/massshatter_3bej6fq7_2026_09_24/`. New script
`scripts/eval/matched_intensity_filter.py` (7 tests), figures and CSVs in
`docs/figures/masster_matched_intensity_filter_test_*`. Tooling: the export's candidate-level
`--min_peak_int` is fixed (entry above); `plot_predictions_hdf5.py`'s CE figures still silently
drop NCE ≥ 70 (37% of this split).
[Finding](findings/masster/masster_matched_intensity_filter_2026_09_24.md).

**Export peak filter fixed for MASSter (2026-09-24); uncommitted.** In
`batch_filter_by_peak_intensity` (`src/fragnnet/utils/script_utils.py`), the peak filter was
gated on `frag_formula_peak_idxs`, the DAG peak -> formula map only FraGNNet batches carry. So
`--min_peak_int` did nothing for MASSter: exports kept all ~2,000 candidates per spectrum.
Now peaks are filtered whenever `min_peak_int > 0`. The formula, joint and node cascade still
needs the map; without it those arrays pass through whole, as before. FraGNNet output is
unchanged. **Behavior change:** `run_save_inference_eval.py` defaults to `--min_peak_int 1.0`
(0.1% of base). With the m/z-aggregated export (entry above), that drops about 89% of peaks
by default: median 924 to 65 per spectrum on MSGYM test. Any analysis that needs every peak must
pass `--min_peak_int 0`. The earlier MASSter exports (cross-eval, ranking decomposition,
per-peak uncertainty) were effectively unfiltered candidate rows. Two export runs of one
checkpoint differ on about 1.3% of spectra (up to 0.015) from GPU nondeterminism in the forward
pass; means agree to 1e-5.

Tests: 7 new in `tests/test_batch_filter_peak_intensity.py`; the pre-fix code keeps 9 of 9
peaks in the key case. The 362 inference/export tests pass, and ruff is clean. A 3-batch
MSGYM export printed the filter stats (88.1% removed, no spectrum emptied, minimum exported
intensity 1.00003). Logged `cos_sim_0.01` is unaffected; 1-2 of 192 spectra move by up to
1.4e-4 with the filter on or off, which is GPU nondeterminism. On the MSGYM test set a 0.1%
cutoff costs −0.0001 on linear cosine; see the
[MSGYM-vs-GLACIER finding](findings/masster/masster_msgym_vs_glacier_released_ckpt_2026_09_24.md#peak-over-prediction-added-2026-09-24).

**MASSter beats GLACIER's released MassSpecGym checkpoint on the benchmark's linear cosine
(2026-09-24, corrected); inference only.** GLACIER outputs sqrt intensity: ms-pred trains it on
sqrt(inten / max), so its output is squared for a linear comparison. On the 9,954-spectrum
simulation test set, on the linear `cos_sim` MassSpecGym requires, computed with our exact
`cos_sim_0.01`: `ll04i00m` 0.5713, GLACIER 0.5455, a gap of +0.026 (99.9% CI +0.009 to +0.044 by
molecule). In sqrt space, where GLACIER trains and ms-pred evaluates, the gap is +0.011 (CI −0.002
to +0.024); per-peak sqrt at a matched peak count gives +0.002 to +0.006. GLACIER's sqrt-space
0.566-0.577 is close to its reported ~0.58. The lead is at high CE on FT (linear +0.050 above
50 eV); in sqrt space GLACIER is level or slightly ahead at CE ≤ 35 and on QTOF. The checkpoint's
3,098 steps per epoch fit training on the simulation train fold only. `ll04i00m` is not our best
recipe (`1imawkx1` has no checkpoint), so the EMA arm's checkpoint is the next one to score.
[Finding](findings/masster/masster_msgym_vs_glacier_released_ckpt_2026_09_24.md).

**MASSter route map (2026-09-24): the encoder's global attention writes the per-molecule
lookup key; inference only.** Step 0 of the memorization-route plan extended the probe to routes
outside the token and ran it on `6h67zps9` and `3bej6fq7`, with 17 paired conditions and
candidates held fixed. Replacing each atom's global-attention message with the batch mean costs
train 2.6x what it costs val (the pools cost 1.2x), cutting the gap from 0.18 to 0.04 while val
falls only 0.08-0.09. The boundary block reads the key (1.46x). The ion/H head is mostly
generalizing, exact mass is weak and `a_charge` precision is null. The checkpoints agree to about
0.01. Consequences:

- `r2_no_gsa` has the strongest evidence of the Step 1 arms.
- `k1` is weaker, since the key is written upstream of the head and `h1`'s allocator dropout was
  rejected.
- Global-message dropout is a new targeted candidate (not built).

[Finding](findings/masster/masster_route_map_2026_09_24.md).

**Memorization literature, three passes (2026-09-24); reading only.** The ranked list maps each
method family onto the route map. The third pass (Consensus database) adds three things. First,
an open decision C: every MASSter run is constant `lr: 1e-3`, and a large initial rate with
decay keeps easy-to-fit patterns (a molecule key) from being memorized first. `d3` gained
+0.0072 on the merged lineage (n=1). The proposal is a separate arm `s1`, not built. Second,
DropKey and a nonparametric VIB as global-message options. Third, a variational sink. Periodic
resets are demoted. MassFormer's pro-global-attention result is the counter-evidence `r2` tests.
[Note](notes/memorization_literature_2026_09_24.md#9-consensus-pass-third-pass-2026-09-24) ·
[change list](plans/masster/masster_memorization_route_plan.md#change-list-2026-09-24).

**W&B checked (2026-09-24): a weight EMA is the first training-side lever to pass on both
lineages; read-only.** Nothing is running. `u2_ema999` (`c8hh9j1b`, unmerged `[M+H]+`/FT)
reaches a 54-63 tail of 0.7392 against 0.7298 (+0.0094). `t2_ema998` (`49ub2p8y`, merged)
reaches a 118-127 tail of 0.7363 against 0.7266 for `c4a_base_s1` (+0.0097). The s1/s2 seed
spread is 0.0007. Both arms beat their control at every epoch. `t2` at 128 epochs also beats the
630-epoch `long_control` best (0.7318). `h1` allocator dropout 0.2 is rejected (−0.0067), and
`h2` width 128 is null (−0.0015). `t2`'s train-metrics control was never submitted, and `t2`
did not upload checkpoints. Next: the offline ranking decomposition and novelty bins on `u2`,
then an EMA + `d3` stacking arm. Whether `ema_decay` becomes the screen default is an open
decision. [Plan read](plans/masster/masster_ranking_transfer_plan.md#tail-read-from-wb-2026-09-24).
The MSGYM GLACIER-parity features arm is a null to slightly negative; see its entry below.

**Per-peak uncertainty from ensemble spread probed (2026-09-24); offline only.** The four f0/f1
raw-val checkpoints (two runs × two epochs) were clustered per peak at 10 ppm. Member votes
predict whether a predicted peak exists in the target: 76% at 4/4 against 23% at 1/4. The spread
is mostly spectrum-level, though. Within a spectrum, at matched intensity, the higher-spread peak
errs more only 58-60% of the time, and per-peak features add +0.007 presence AUC over
spectrum-level confidence (0.793 → 0.800). Raw μ ± 2σ covers 76% and is worst at NCE < 20, so
intervals need held-out (conformal) calibration. 14% of squared error is in target peaks the
model does not show, and replicate noise is about 10%. Uncertainty must therefore come from
disagreement or held-out residuals, never from a head trained on training residuals. Next: a
K = 4-5 seed set to see whether the per-peak signal grows with K, and a held-out error model with
candidate descriptors. See [the finding](findings/masster/masster_per_peak_uncertainty_2026_09_24.md).

**MASSter memorization-route Step 1 built (2026-09-24); ran the same day, read above.** The probe found
the per-molecule lookup in the boundary block. But every candidate input comes from the same
globally attended atom embeddings (GINE_GSA), so a blocked input can re-form the key from the
others. Step 1 has five arms, each a same-code control `c0` plus its own keys, in
`config/masster/mh_ft_memorization_route_2026_09_24/`:

- `r0` removes the boundary block (a rerouting test);
- `r1` gives the cut term a one-layer local encoder with no global attention;
- `r2` sets `mol_gnn_type: GINE`;
- `k1` adds dropout 0.2 on the key-forming layers;
- `m1` adds a per-molecule memorization sink, zero-initialized and training-only, so val scores
  the shared model.

Seven default-off template keys. At their defaults the model is unchanged, and the new modules
use a forked RNG. 27 new tests pass, plus the 413 existing MASSter tests. Two-epoch smokes of
every option pass, and the `m1` checkpoint reloads strict. `c0` also logs candidate coverage.
Generate on Fir with `submit_masster_mh_ft_memorization_route.sh all` (`fir:h100-3g.40gb`).
Next: build Step 0 (the route map: encoder `gsa_out`, residual versus ion/H head, per descriptor,
collision census) while they train, since reading the arms needs it.
[Plan](plans/masster/masster_memorization_route_plan.md#step-1-implementation-and-validation-2026-09-24).

**Intensity displacement, Phase 0 read (2026-09-24); offline, no training.** On the f0/f1
unmerged `[M+H]+`/FT exports:

- **Most of the misplaced intensity is shared bias.** 72.5% is common to both runs. After the
  replicate target-noise floor, the reducible shared part is 67% (val) and 63% (test). Variance
  reduction such as EMA can reach at most the 28% run-specific share.
- **It is not neighbour confusion.** One-loss moves look enriched (1.75x) only against a naive
  null. Against a null that keeps mass on the real peaks, every class is within 0.90-1.08x. The
  wrong base-peak winner is a real peak 94% of the time and is no closer to the true base than a
  random real peak.
- **The one structured part is the precursor/parent fallback below NCE 60.** At NCE ≤ 20, 0.64 of
  nested mass sits on the heavier member, against 0.19 for the null.
- **Formula-level loss identity is complementary there.** A formula-only learner stacked with the
  model lifts precursor-versus-contender AUC 0.798 → 0.829 (val) and 0.807 → 0.838 (test).

Class-targeted arms dropped.
[Finding](findings/masster/masster_intensity_displacement_phase0_2026_09_24.md) ·
[plan](plans/masster/masster_intensity_displacement_plan.md).

**Phase 1.4b, bond-local learner (2026-09-24): gate G1 for arm 2A not passed.** Identity-blind
GBDT/MLP learners on bond-local features were fitted on 12,055 train molecules. Stacked onto
f1's fragment scores at peak level and cross-fitted between val and test, they add
+0.0022 ± 0.0003 (val) and +0.0025 ± 0.0003 (test) group cosine over a fitted temperature.

- **Beyond a stronger control.** With per-NCE temperature and precursor offsets as the
  baseline, the gain is +0.0017 / +0.0028.
- **Where it helps.** The gain is largest on novel molecules (+0.0032 / +0.0037) and at
  NCE ≤ 40, and near zero above NCE 60.
- **What it is not.** It is not precursor survival (0.0000) and not the bond-type × CE trend
  (+0.0006).
- **Alone, the learner orders bonds worse than the model.**

The result is in the grey zone between FAIL (+0.002) and PASS (+0.005), so 2A is not submitted.
One offline retry is open: a peak-level-trained learner with hashed environments.
[Finding](findings/masster/masster_bond_local_learner_2026_09_24.md).

**2026-09-23 run results recorded, plus two corrections (2026-09-24).** Read-only W&B and code
reading; the only non-doc edit is a config comment.

- **Fragment-marginal selection is CLOSED NULL.** Marginal K128 scores 0.0000 from U256 and
  −0.0016 from U512, and loses to mask-order U128. The August U128 penalty does not reproduce:
  mask-order U128 (`f5`) is +0.0007 at 0.52 against 0.85 min/epoch, so 1.65x faster. U64 costs
  −0.0090. [Plan](plans/masster/masster_fragment_marginal_selection_plan.md#tail-read-2026-09-24-marginal-selection-fails-and-u128-is-a-free-speed-up).
- **The feature bundle is worth 0.006, and no single component clears the gate.** `no_bbe`
  −0.0024, `no_acharge` −0.0025, `no_charge_feat` −0.0012, `no_mass_feats` +0.0012,
  `bundle_off` −0.0060.
  [Campaign read-out §8](findings/masster/masster_campaign_readout_2026_09_21.md#8-the-bundles-leave-one-out-arms-read-at-the-128-epoch-screen-added-2026-09-24).
- **Correction: `boundary_bond_encoder` was never removed on 2026-07-13.** That removal was
  FraGNNet's uncommitted `BoundaryBondEncoder`. The MASSter key dates from 2026-09-20, so the
  `no_bbe` config's revert rule has no premise, and the encoder default is an open decision.
- **The charge head is inert in `c4a`.** Its 263,169 parameters are trained by the proposal
  loss and change neither the kept fragments nor the spectrum. §7.1's −0.01 gradient-connection
  cost is a convergence delay: the connected run reaches its control by epochs 610–629.
  [Head-input asymmetry §8](findings/masster/masster_head_input_asymmetry_2026_09_22.md#8-additions-2026-09-24).
- **The memorization probe's boundary block is the `CutBondEncoder` output.** The 0f99e7e4
  dedup does not change it.
  [Probe](findings/masster/masster_memorization_route_probe_2026_09_23.md).

Open decisions: whether to rebase screens on U128 for speed; whether `boundary_bond_encoder`
stays on; and which merged boundary-block arm to run. Any such arm needs
`compute_train_auxiliary_metrics: true` and `upload_checkpoints: true` so the probe can be run
on it.

**MassSpecGym GLACIER-parity features arm: RAN, no gain (W&B read 2026-09-24).** `ll04i00m`
reaches a 118-127 val `cos_sim_0.01` tail of 0.5669 against 0.5695 for `1imawkx1` (−0.0026),
and is below it at 101 of 128 epochs. Best checkpoints: 0.5705 (epoch 60) against 0.5754
(epoch 62). Both runs peak near epoch 60 and plateau or drift down after it, so this MSGYM recipe
is past its best by mid-schedule. `1imawkx1` has no uploaded checkpoints; `ll04i00m` does. Both
are n=1 with no MSGYM seed replicate. Overfit timing in `1imawkx1`: val proposal loss is lowest
at epoch 7 and val `coverage_weighted_spec_ce` (the primary loss) at epoch 14 (−1.793, ending at
−1.580). Val cosine peaks at 62, while train loss falls to epoch 127. The CE rise is not coverage:
`val_wrecall` is flat at 0.873-0.877 from epoch 8, and the conditional CE rises by the same
amount. `val_wprecision` climbs from 0.59 to 0.71, which is consistent with predictions
sharpening; a temperature fit on a checkpoint would test that. NIST `3bej6fq7` shows the same
shape much more mildly (lowest at 27, then +0.027). Data lever, measured 2026-09-24: the
`ms_gym_all_1_5` train fold adds 9,313 `[M+H]+` molecules (35,731 spectra, +76% over the
benchmark's 12,304). None of them is in the benchmark val or test sets by 2D InChIKey. Only 21
of them have a CE; missing CE becomes −1 and, under `abs-sines`, a dedicated learned extra
embedding. **Scope decision (user, 2026-09-24): benchmark subset only; no extra MassSpecGym
data.** How the ms-pred baselines handle missing CE on MassSpecGym (code at `318bb7b`,
`iceberg_2.0` branch at `74b5103`, and the papers, read 2026-09-24):

- **ICEBERG 2.0, as published:** known-CE spectra only, for training and testing (supplement
  §S2, p. 8: "in line with the setting used by FraGNNet and the MassSpecGym 'spectrum simulation
  challenge'"). The branch also has an unreported `msg_allev` config that reads pre-built
  `[imputed]` eV spectra. No imputation code ships with it.
- **ICEBERG 2.1 (`main`):** `msg_simulation` keeps the `simulation_challenge` rows whose CE
  parses as a number (119,029 expected) and asserts that none are imputed. `msg_all`, for
  retrieval, imputes by pseudo-labelling rather than interpolating
  (`run_scripts/iceberg/msg_all/04_impute_missing_collision_energies.py`). A known-CE model
  predicts each missing-CE spectrum at NCE 5-150 in steps of 5, the NCE with the best
  entropy-similarity match wins, and the model retrains on real plus pseudo labels.
- **GLACIER:** no imputation. `configs/glacier/joint_train_msg.yaml` trains on the base `msg`
  dataset, and a NaN CE takes a fixed all-zero CE embedding (`joint_model.py`,
  `collision_embed_merged`). The paper gives 231,104 spectra and 31,602 molecules, and says
  neither how missing CE was handled nor which test rows its similarity numbers cover.

Our raw `MassSpecGym1.5.tsv` has no `[imputed]` labels. All 119,029 `simulation_challenge` rows
carry a numeric CE; 109,358 other rows have none. Our processed benchmark has 118,991 of the
119,029 rows. **EMA arm built,
not submitted:** `..._fast_batch512_cuda_ema999.yml` is the base plus `ema_decay: 0.999`
(`upload_checkpoints` on). Its resolved diff against the base is that key plus naming, and the 5
EMA callback tests pass. No local smoke was run; both the key and the config have trained on
Fir. This is also the first EMA run with LR decay, since both NIST wins used a constant LR. The
original preparation note follows.
**Prepared (2026-09-24).**
`config/masster/massspecgym/msgym_benchmark_1_5_unmerged_dual_ce_fast_batch512_cuda_glacier_feats.yml`
is the running MSGYM config (`1imawkx1`: val `cos_sim_0.01` 0.568 best at epoch 38/128) plus the
four GLACIER-parity features that were never trained anywhere: `a_fused_ring`, `b_ring_size`,
`b_fused_ring`, and `c_nl_composition` (neutral-loss element counts, GLACIER's formula term).
`upload_checkpoints` is on. The resolved diff is exactly those keys plus naming. The model grows by
11k parameters (intensity token 1,285 to 1,303 inputs). All 16,967 MSGYM molecules featurize, and
a 2-epoch smoke on a 256-spectrum subsample trains (batch 32; the production batch 512 does not fit
the local 8 GB card). Read against `1imawkx1` on the same val set, then on the simulation test set.
Data note: `ms_gym_benchmark_1_5` is exactly MassSpecGym's official simulation-challenge set
(`simulation_challenge == True`, 119,029 `[M+H]+` spectra, all with CE). The other 112,075 spectra
lack CE on 97.6% of rows. GLACIER trained on all 231,104 for retrieval and does not say which test
population its 0.58 cosine covers. Generate on Fir with `setup_job_cc.py` (`fir:h100-3g.40gb`).

**Run-to-run variance is epistemic (2026-09-24); offline only.** On the f0/f1 unmerged
`[M+H]+`/FT val exports, agreement between the two runs falls from 0.94 to 0.75, and the
ensemble gain rises from +0.005 to +0.034, as a val molecule's nearest-train-neighbour Tanimoto
drops from above 0.8 to below 0.3. This holds within every NCE bin. Adding the 17,006 NIST23-only
`[M+H]+`/FT molecules to training is estimated at about +0.018 (correlational). See
[the finding](findings/masster/masster_variance_novelty_2026_09_24.md). Next checks: a pure
seed pair, and whether removing the boundary block (`n1`, not built) raises run-to-run agreement.
A mixture-of-experts version is a later option; the finding notes the design constraint.

**Formula-only baseline measured (2026-09-23); offline only.** On the f1 (`6h67zps9`) unmerged
`[M+H]+`/FT exports, a predictor that sees only formulas and NCE reaches 0.38-0.40 val cosine
(lookup) / 0.46 (learned) against the model's 0.76. Blending it into the model adds +0.005, most
of it above NCE 60, so the transferable formula signal is small and mostly already in the model.
See [the finding](findings/masster/masster_formula_only_baseline_2026_09_23.md). A
formula-first token without the boundary block (`n1`/`n2`, not built) should therefore be judged
as removing a memorizing input, not as adding formula information.

**MASSter MSGYM config moved to the best measured recipe (2026-09-23); smoke-tested, nothing
submitted.** `config/masster/massspecgym/msgym_benchmark_1_5_unmerged_dual_ce_fast_batch512_cuda.yml`
had never been trained. It now resolves to `c4a_base_s1` plus `a1_prenorm` plus the `d3` cosine
schedule (warmup 1000, floor 1%), with MSGYM data, `[M+H]+` only, no CMF, `nce_max: 300`. It
runs 128 epochs of 194 steps (`lr_total_steps: 24832`, was 174/290 constant LR). The three
2026-09-20 defaults are pinned. Resolved-config diffs against `a1`, `d3` and `c4a_base_s1` show
only those keys. Unmerged is kept, since the MSGYM benchmark scores per spectrum. A 2-epoch smoke
fit on a 256-spectrum subsample finished (9.2M parameters, loss falling), and the checkpoint
records `pre`/`cosine` and ends at 1e-5. Caveats: `a1`+`d3` together is untested, `d3` is n=1,
and there is no MASSter MSGYM baseline to read it against. The unmerged NIST20 parent
`6ks7v8ce` took about 5 h for that step count on an L40S (0.73 s/step, 26.6 GiB peak).
Generate on Fir with `setup_job_cc.py` (`fir:h100-3g.40gb`, `j=<job_name>` for resume).

**Shortcut-reduction screen prepared (2026-09-24); RAN, read in the top entry.** The shortcut audit
(overfit read-out §7) now leaves one measured shortcut on unmerged `[M+H]+`/FT: a per-molecule
allocation lookup. Exposure is flat (Spearman 0.058), the precursor prior is only a fallback on
unseen molecules, and the fingerprint is under 11%. Identity reaches every head input, so the arms
limit the allocator or remove optimizer noise instead of blocking one input: `h1` intensity-transformer
dropout 0.2, `h2` width 128 (head 2.46M → 0.97M), `u2` EMA 0.999 (new `ema_decay` key and
`ExponentialMovingAverage` callback), all one key off `f0` with the existing run `3bej6fq7` as
control (tail mean 0.7298). The merged `t2` EMA arm stays. The boundary-block dropout arms `t1`/`u1`
were dropped for rerouting; the key stays, off by default. Read the val and train-sample ranking
gaps and the novelty bins offline, not only the tail mean. Generators:
`submit_masster_mh_ft_ranking_transfer.sh` and `submit_masster_regularization.sh transfer`.
[Plan](plans/masster/masster_ranking_transfer_plan.md).

**MASSter still has the ranking problem (2026-09-23); offline, on the cross-eval exports.**
For `6h67zps9` on its cleaned target, the cosine gap is coverage 0.014, **ranking 0.121** and
calibration ceiling 0.104. Ranking is the part no order-preserving rescale can fix. The base peak
is hit 56.7%, under-ranked 41.6% and uncovered 1.7%, and test replicates to 0.01. Ranking doubles
with CE, while calibration is flat. The low-CE precursor default FraGNNet showed is present:
0.88 vs 0.42 base-peak hit at NCE ≤ 20. Unlike FraGNNet, the predictions are slightly too sharp,
and a global `P^0.75` adds +0.007 on test for free. Neither averaging nor target cleaning moves
the ranking gap below about 0.11. The lever is a ranking objective on the candidate
distribution; none exists in `src/`. **Update:** on a 25.6k train sample the ranking gap is 0.011
and the base-peak hit rate 84.5%, bin by bin. The ranking error is a generalization failure, so
a train-set ranking loss has little to fit. Target transfer instead: boundary-block
regularization, variance reduction, and priors that generalize.
[Finding](findings/masster/masster_ranking_decomposition_2026_09_23.md).

**`mh_ft_clean` cross-eval run (2026-09-23); target cleaning is a CLOSED NULL, and run-to-run
variance is the largest lever measured; inference only.** Both checkpoints were scored against
both targets on val and test: `6h67zps9` (f1, cleaned target) and `3bej6fq7` (f0). On the raw
target, f1 − f0 is +0.0001 (val) and +0.0007 (test), inside ±0.003. The W&B +0.0054 is the easier
target, +0.0048 to +0.0052 for either checkpoint. So the ~0.006 the 0.73-vs-0.80 cross-tab gives
`filter_subformula_peaks` is denominator too. The two runs disagree per spectrum, with a median
prediction cosine of 0.94 and a 10th percentile of 0.65, and averaging their predictions adds
**+0.020** on both splits, rising with CE. Two adjacent checkpoints of one run add +0.008.
**Averaging those two checkpoints' weights beats each run's best checkpoint by +0.0029 / +0.0036
with no training.** Next: an EMA/SWA weight average over the last epochs (nothing in `src/`
does this) and a `d3` replicate, then a seed pair to price the between-run share.
[Finding](findings/masster/masster_target_cleaning_cross_eval_2026_09_23.md).

**MASSter encoder-simplification arm `g1` read (2026-09-23): a REJECTION.** Read-only W&B
query. `xb9ic2pp` replaces the control's 2 GINE_GSA layers and 3-block `BondFocusedBreakHead`
with one 5-layer encoder and an endpoint bond head, and turns the boundary `CutBondEncoder` off
(9.22M → 7.69M parameters). Its 118–127 tail mean is **0.7110**: −0.0160 against the 0.7270
control, and −0.0136 against `c4a_no_bbe`, which differs from it only in the two depth keys. It
is below both controls at all 128 epochs, and being 4% faster per epoch does not rescue it. The
loss is intensity fit: primary loss is worse on train and on val, while the tail val proposal
loss is level. The proposal reads the encoder detached, so the bond head cannot change what
trains the encoder, and the likely cause is the 5-layer encoder. The two keys are not
separated; the one-key arms that would isolate them are `masster_bond_num_layers: 0` and
`mol_num_layers: 5`. The refinement head also reaches a lower val proposal loss than the
endpoint head by epoch 35, then drifts up; equal tail loss does not make it pure train fit.
[Finding §5](findings/masster/masster_architecture_arms_epoch128_2026_09_22.md#5-encoder-simplification-arm-g1-added-2026-09-23).

**`CutBondEncoder` now encodes each directed cut bond once (2026-09-23).** Both callers,
MASSter's `boundary_bond_encoder` and FraGNNet's `cut_bond`, ran the encoder on every
(fragment, cut bond) row, although its output depends only on the two endpoint embeddings. That
is 9.8x repeated rows on a 24-molecule NIST20 MASSter batch and 29x on the FraGNNet test DAG.
The new `encode_cut_bond_pairs` in `base_model.py` encodes each distinct (inside, outside) pair
once and gathers the rows back. Parameters are unchanged, so checkpoints load. Verified against
the old code on the production-shaped MASSter model: predictions within 9.5e-7, every
non-negligible gradient within 4.2e-7 relative. The FraGNNet `cut_bond` test model is
bit-identical. Encoder forward FLOPs fell from 20.1 to 2.1 GFLOP (MASSter total 45.9 to 27.8).
Expect little wall-clock from it: removing the encoder outright (`c4a_no_bbe`) saved only 6%
per epoch. The model function is unchanged, so this does not touch the boundary block's
[memorization](findings/masster/masster_memorization_route_probe_2026_09_23.md). Tests
in `tests/test_cut_bond.py` and `tests/test_masster_candidate_features.py`, including
per-row equivalence checks for both models that catch a planted orientation bug.

**MASSter memorization located in the intensity head's boundary block (2026-09-23); inference
only, no training.** The pooled-embedding corruption probe gained `features` and `set` scopes and
ran in eval mode on two production checkpoints, `6h67zps9` and `3bej6fq7` (unmerged `[M+H]+` FT,
not the merged `c4a_base` lineage). Both give a train/validation cosine gap of 0.19 and a
coverage difference of 0.007. At matched validation damage, corrupting the boundary block leaves
a 0.022 gap, against 0.059-0.086 for the fragment/complement pools. The pools and the candidate
descriptors act as generalizing features. The candidate-set fingerprint carries under 11% of the
gap. So the null `e2`/`e3` `candidate_atom_dropout` arms regularized the wrong columns. Next:
extend that dropout to the boundary pool, or add dropout of the whole block, on the merged
128-epoch screen with train metrics on. Not answered: the data-side routes, and whether the
encoder or the head's weights hold the identity.
[Finding](findings/masster/masster_memorization_route_probe_2026_09_23.md).

**Proposal early-stop arms read; both nulls, and the family is closed (2026-09-23).** Read-only
W&B query, nothing changed but the docs. Tail mean 118-127 against the 0.7270 control: `p1`
freeze@32 `s5i5qqlz` **+0.0010**, `p3` proposal wd 0.1 `w1dj2ahw` **+0.0020**. `p2` is not worth
submitting. The useful by-product is that `p1` measured the shared encoder drifting under
frozen proposal heads. Train proposal loss rises in every block after epoch 32 (+0.082 in
total), and about a third of the control's late val-proposal degradation survives the freeze.
That means a clean freeze needs `separate_proposal_encoder`. It does **not** mean the shared
encoder costs cosine. Proposal loss and cosine are decoupled across the four runs, and the one
same-seed, one-key separate-vs-shared test (`esb2fqhg`/`d7s0yb72`, 2026-08-21, old merged
lineage, never recorded before) is **−0.0008**. A separate-encoder arm is a one-key 128-epoch
test with a null prior, ranked behind the `a1_prenorm` and `d3` replicates.
[Plan § Results](plans/masster/masster_proposal_early_stop_plan.md#results-p1-and-p3-at-the-128-epoch-screen-2026-09-23).

**SMARTS usage audited end to end (2026-09-23); read-only, nothing changed but one new probe.**
No production model uses a SMARTS rule. In MASSter, the eleven compound ring-cut rules add
only a merged neutral byproduct (chance-floor intensity); their use is forcing a 4-bond mask.
The neutral-loss rules are the real new chemistry, bounded at about 0.1% of NIST20 intensity.
Past FraGNNet prepass verdicts measured `crf12`/`crf13`, which make up 99% of its partitions.
Also: a dead sulfoxide rule, `uniquify` dropping anthraquinone partitions, and stale
`smpFalse`/`ringmerge` stats files. Next is the model-free coverage check in §5.1 before any
CUDA port.
[Finding](findings/smarts_usage_audit_2026_09_23.md).

**MASSter fragment-marginal selection implemented (2026-09-23); nothing trained or submitted.**
A U128 cap loses cosine because it truncates the pool in mask order, and the proposal loss gives
zero gradient to peaks only the cut-off fragments explain. The new opt-in
`masster_fragment_selection: marginal` keeps the forced precursor/single cuts plus the top
`masster_selected_fragment_target` fragments by closed-form release probability, and scores
the proposal loss on the whole enumerated pool. Default `mask_order` is unchanged; the full suite
passes (4,271, after two review fixes). Three arms against `c4a_base` are prepared: `f1` mask-order U128, `f2` marginal
K128 from U256, `f3` from U512. Generator: `scripts/manual_slurm/fir/submit_masster_fragment_selection.sh`.
Reading rules and open points:
[plan](plans/masster/masster_fragment_marginal_selection_plan.md).

**FraGNNet FT contender-pair Stage A prepared for Fir (2026-09-23).** Configs and a generation-only
script; **nothing submitted.** Four arms from `fraggnn_d3_ma_mi_nist20_amolc_all.yml` (the
`jnokddr9` config): `c0` original sampler, `c1` `ce_context` sampler control, `p1`/`p2`
`ce_pair_loss_weight` 1.5 / 6.0. The weights come from a measured initial gradient ratio
g = 0.0344 on ~406-spectrum steps (5% / 20% targets). Two things change the matrix from the plan's
reading. First, `target_rel_ints_thresh` (required by the pair loss) also filters the
**validation** target, so all four arms carry 0.01 and `jnokddr9` is not the control. Second, the
original group sampler shows only 138,875 of 287,721 train spectra per epoch, so the context arms
run 48 epochs to match c0's 100 on spectra seen. The primary read is p1/p2 vs c1. Details, open
gate items and costs:
[plan § Stage A preparation](plans/fragnnet/ft_contender_pair_minimal_training_plan.md#stage-a-preparation-2026-09-23).
Generator: `scripts/manual_slurm/fir/submit_fraggnn_ce_pair.sh` (run on Fir; needs the DAG tarball
and `wheels/` there).

**`a5_ce_pair_bias` and the full-length `wd10` arm prepared for Fir (2026-09-22).** Config and
script only; **nothing submitted, no GPU used, no run launched.** Two arms, three jobs.

**`a5`** needed no config change (already 128/128). `submit_masster_attention.sh` gained an
**`a5` stage** so it can be generated without `a4`, which the epoch-128 read argued against in the
same motion that raised `a5`. `a5` resolves to **three** keys against `c4a_base_s1`, so it must be
**read against `a3_full_massbias` (`4fg0x5vw`)**, not the control — `a3` holds
`attention_type: full` and the mass pair bias without the CE term, and `a5` − `a3` is the only
contrast that isolates `intensity_ce_pair_bias`. `fir:h100-3g.40gb`, 3 h.

**`wd10` cost more than it looked and the reason is memory, not wall clock.** The question is the
train/validation gap, which needs a **train** cosine, and `compute_train_auxiliary_metrics` is
**false** in `c4a_base_s1/s2` — so the existing control supplies a matched val cosine and no
matched train cosine at all. `ctrl_a`/`ctrl_b` do have it on but predate the 2026-09-20 default
flip, so they are a different architecture. Two new/edited configs, both never run:
`..._c4d_wd10.yml` moved 630 → **128/128** with the metric on, and a new
`..._c4a_base_s1_trainaux.yml`. Resolved diffs verified: control − base = that key alone,
**arm − control = `weight_decay` alone**. Same seed 397623, so the control's val cosine is also a
third same-seed replicate of `c4a_base_s1`. Generator:
`scripts/manual_slurm/fir/submit_masster_regularization.sh wd10`.

**Both wd10 jobs are `fir:h100-80gb`, not the 3g.40gb slice, and that doubles their price (12.2
vs 6.1 each).** Measured from the W&B system stream on 2026-09-22: the metric takes peak GPU from
**30.4 GiB** (`c4a_base_s1`, off) to **36.2–37.2 GiB** (`ri3zm215`, `kalc4mzo`, on), against a
3g.40gb slice's **37.25 GiB** — under 40 MiB of headroom before the CUDA context. Slice size does
not affect the metric, only throughput, and every read here is at matched epoch. The same
measurement clears the wall-clock worry: the key costs **+6%** per epoch (0.91 vs 0.86 min/epoch),
so 128 epochs is ~1.95 h in a 3 h box. **The 2026-09-19 arms died at epoch 64-65 because they had
a one-hour box, not because of this key** — that screen was ~10% exposure.

Two cautions carried into the read. `kalc4mzo` is **not** this arm's predecessor: it ran
2026-09-19, before the three candidate/h-transfer defaults flipped, so re-resolving its config
today inherits defaults it never trained with. And the prior does not favour the arm — **three
arms have now narrowed the gap and lost** (`drop03`, `b3_wd_groups`, `e2_cc_atom_drop01`); it is
run to close the question, not because it is expected to win. Also noted while copying:
`c4a_base_s1.yml` carries a `train_aux_metrics` W&B **tag** while the key is `false`, the wrong
tag the overfit read-out already caught; the copy's tag was corrected, the original left alone so
it still matches its indexed run.

Validation: both new configs parse and resolve; `max_epochs: 128` and the train-metric key
asserted by the generator; `bash -n` clean on both scripts; resolved one-key diffs verified
through `load_config`. Job generation itself must run on Fir — the wheel preflight needs
`wheels/`, which is Git-ignored and absent from this checkout.

**`a2_full_attention` and `b1_bond_sum` read; §1's capacity confound resolved, and the family's
reproducibility floor measured (2026-09-22).** Read-only W&B query, no GPU, nothing changed.
Both lead-pair arms finished at 127 epochs on the same screen (tail mean 118-127, control
**0.7270**, gate ±0.005), so the question the previous handoff left open is answered.
**`b1_bond_sum` is a null at −0.0013** — the degree-normalized mean is not costing anything, do
not re-run it. `a2_full_attention` lands at **+0.0030**, and as the topology-only control it
splits `a6_ce_film`'s +0.0063 about evenly: **+0.0030 from `attention_type: full` itself
(at 41% fewer intensity-head parameters) and +0.0033 from CE-FiLM**. Neither half clears the gate
alone. The sharper result is `a3`: **the pairwise mass bias is an exact null (−0.0002) at matched
topology**, so all of `a3`'s earlier +0.0028 was the attention change, and the review's
mass-blindness item is *not* answered by that term. `a1_prenorm` (+0.0056, one key, no capacity
change) is still the only clean single-key PASS and is now also the cheaper of the two leaders.

The accidental duplicate finally pays for itself. `xyessyn1` and `cgomm8v4` are the same config
and the **same seed** (`deterministic: False` in both — the `deterministic` W&B *tag* on these runs
is wrong), and they diverge from epoch 0 and never reconverge. **The floor depends entirely on the
statistic:** 0.0025 mean |Δ| on a single epoch (0.0053 at epoch 64), 0.00036 on the tail mean of
118-127, 0.00001 on best ≤127. That **validates the 118-127 tail mean every verdict in this family
uses** — its run-to-run floor is tighter than the 0.0007 cross-seed control spread — and rules out
single-epoch reads, which carry more noise than five of the eleven arm effects on record. It also
puts a much coarser floor under the *mechanistic* columns: **0.011 on the loss drift and 0.014 on
the train/val loss gap**. The turn *epoch* is reproducible (78 vs 78 at fixed seed, 52 vs 66 across
seeds), so `a1_prenorm`'s turn at epoch 120 stands; but the "2.6x drift cut" is ~1.7 floor-widths
and should be read as directional, and the drift ordering among `a3`/`a6`/`b2`/`a7` is inside the
floor. Full tables in [epoch-128 read-out §3](findings/masster/masster_architecture_arms_epoch128_2026_09_22.md).

**Next.** `a5_ce_pair_bias` and `a4` are what remain of review item 1, and a seed replicate of
`a1_prenorm` is now clearly worth a box — it and `d3` are the two live candidates and both are n=1.
Promote nothing on a single-epoch read; the gate on tail-mean comparisons could be tightened well
below ±0.005, but not on one duplicate pair.

**Proposal warm-up read at the matched screen and rejected as the alternative (2026-09-22).**
Asked whether to freeze the proposal branch or warm it up and lean on weight decay. The three
`c4c_pwarm` arms had never been read at the 128-epoch screen — they ran 630-epoch configs and
were killed near epoch 205, so their summaries were last-logged values at unmatched epochs.
Tail mean of epochs 118-127 against the same 0.7270 control (gate ±0.005): `e2` **+0.0008**,
`e4` **+0.0012**, `e15` **−0.0096** — two nulls and a rejection, non-monotone, and the horizon
does not confound it (all carry `lr_schedule: false` at constant `lr: 1e-3`, all seed 397623).
On top of that the key un-detaches the proposal branch for the whole run whenever the encoder is
shared, which is every `ma_mi_merged` config, and that rewiring measured −0.0113@100 / −0.0132@400
on its own. Warm-up also treats the start of the run where the defect is at epochs 26-47. Read-only
W&B query, no GPU, nothing changed by that read. Recommendation recorded in the plan: run `p1`
(freeze@32) and `p3` (proposal wd 0.1), hold `p2`, drop warm-up.

Proposal early stop and proposal-only weight decay implemented (2026-09-22, later the same day).
**Code and config; no run launched, no GPU used.** The
[overfit read-out §3](findings/masster/masster_overfit_readout_2026_09_22.md) records that the
validation **proposal** term degrades +0.06 to +0.14 and **turns at epochs 26-47** while the
spectrum term keeps improving to 52-120 — two branches on one training schedule. Under
`proposal_selection_only: true` the proposal reads detached inputs, so its loss updates the
proposal heads and nothing else, and those heads are **44.8% of the parameters** (4,128,642 of
9,217,181 at `c4a_base_s1`). Two template keys now let them stop or decay on their own, both
default `null`: `masster_proposal_frozen_from_epoch` freezes the proposal-only modules from a
chosen epoch, and `masster_proposal_weight_decay` gives them their own decay through a new
`_weight_decay_overrides` hook that leaves `SpectrumPL.configure_optimizers` single-group when
nothing opts in. Neither touches `precursor_encoder` or `context_encoder`; a forward/backward test
asserts the proposal objective updates **nothing** past the boundary while the spectrum objective
still reaches the intensity head and both encoders. Three one-key 128-epoch arms are written and
verified in `config/masster/ma_mi_merged/proposal_early_stop_2026_09_22/` (`p1` freeze@32,
`p2` freeze@64, `p3` proposal wd 0.1) and **none has been submitted**. Two limitations belong in
the read and are in the [plan](plans/masster/masster_proposal_early_stop_plan.md): the shared encoders
keep training, so the proposal *function* still drifts (a static proposal needs
`separate_proposal_encoder`, off everywhere), and freezing also puts the heads in eval mode, which
drops their `head_dropout: 0.1` — a second, unseparated change that `p3` does not carry. The same
read-out predicts these may be nulls, since coverage is already flat train-to-val; read
`val_wrecall` next to cosine. Validation: full `pytest tests -q` clean, `ruff check` clean on the
four changed Python files, Black deliberately not run (those files were already not Black-clean at
`HEAD`). No run launched.

Feature and escape arms read (2026-09-22, later the same day). **The plans' "nothing trained"
and "No production run launched" lines were stale — the arms had been submitted, and six of them
now have verdicts. All six are nulls.** Read-only W&B query, no GPU, nothing changed in the tree.

**The five candidate-token feature arms finished at 127 epochs**, on the same screen and against
the same control as the architecture arms (tail mean of epochs 118-127, control **0.7270**,
replicate spread 0.0007, gate ±0.005): `e4_cut_adjacency` **+0.0032**, `e5_cut_bond_plus_adjacency`
+0.0021, `e3_cc_atom_drop02` +0.0002, `e2_cc_atom_drop01` −0.0009, `e1_cut_bond_feats` −0.0010.
**Not one clears the floor.** Adjacency is the only arm with a sign, which is the direction the
[descriptor measurement](findings/masster/masster_candidate_cut_descriptors_2026_09_22.md) predicted
(0.489 residual ambiguity against 0.000 for ring-sharing), so the mechanism argument survives and
the effect does not. Two patterns repeat from the architecture read: **stacking costs** — `e5` =
`e1` + `e4` scores 0.0011 *below* `e4` alone, as `a7` scored below `a1` and `a6` — and **narrowing
the loss gap is still not the objective**: `e2` has the narrowest train/val primary-loss gap of the
seven runs (0.5719 vs a 0.6191 control mean) and the lowest cosine of the five, the third instance
after `b3_wd_groups` and `drop03`. Coverage is flat again (`val_wrecall` spread 0.0006).
`e6_ring_topology_cut_bond` was never submitted. *(Corrected 2026-09-27: it ran, `kc04u2cd`,
+0.0015, null.)*

**The intensity escape is an exact null at n=2 vs n=2.** `e1_escape` against `e0_independent`, two
seeds each on the `batch256x2_lb32` base, read at their common epoch 118 while still running:
0.7225 against 0.7225, **Δ +0.0000**, seed spreads 0.0001 and 0.0003. Re-read at completion, but
nothing in the trajectory suggests a late separation.

**Two caveats worth carrying forward.** (1) Each e-arm differs from `c4a_base_s1` by **eight**
resolved keys, not one — seven new keys resolving to `False`/`0` plus three inert `spec_params`
additions from the above-precursor filter. All are no-ops at those values so the comparison holds,
but check whether `spec_params` participates in any dataset cache identity before the next such
arm: new inert keys in a hashed dict are how a "one-key" arm quietly changes its inputs.
(2) **`a2_full_attention` was submitted twice at the same seed 397623** (`xyessyn1` on `fc11018`,
`cgomm8v4` on `fc10906`) — a duplicate, not a replicate, costing a box that `a5_ce_pair_bias` or an
`a1_prenorm` replicate could have used. It does accidentally measure the run-to-run reproducibility
floor, which is worth having given how much weight the 0.0007 control spread carries.

`b1_bond_sum` (`3bqnobq1`) and `a2_full_attention` launched at 128/128 epochs and were at epoch 20
when this was written — too early to read. Numbers, method and confound checks in
[epoch-128 read-out §2](findings/masster/masster_architecture_arms_epoch128_2026_09_22.md). Banners
corrected on the [cut-set plan](plans/masster/masster_cut_set_features_plan.md) (now SCREENED, ALL NULL)
and the [escape plan](plans/masster/masster_supervised_escape_plan.md) (now IMPLEMENTED, SCREENED NULL).
Validation: relative links resolve, Ruff not applicable, no test run and none required for a
documentation change. No run launched.

Architecture arms read at the matched 128-epoch screen (2026-09-22). **Documentation only; a
read-only W&B query, no GPU, no code, config or run changed.** This closes §1 of the
[architecture review todo](plans/masster/masster_architecture_review_todo.md), which was the blocking
item in the previous handoff: the six arms killed at the wall clock between epochs 179 and 210 now
have verdicts against both `c4a_base` replicates at matched epoch. Numbers are the tail mean of
epochs 118-127 against a control mean of **0.7270** (replicates 0.7266 / 0.7273, spread 0.0007),
gate ±0.005 — full method and robustness checks in the
[epoch-128 read-out](findings/masster/masster_architecture_arms_epoch128_2026_09_22.md).

**`a1_prenorm` +0.0056 PASSES and it is the clean one** — parameter-identical to the base, one key,
and the first such arm in this family to clear the floor. It is also the strongest mechanistic
result in the set: it moves the validation primary-loss optimum from epoch 52/66 to **120** and
cuts the post-turn drift **2.6x** (+0.0121 from its own optimum against a +0.0318 control mean),
which is exactly what the review predicted for post-LN under a constant unwarmed LR.
**`b2_identity_residual` is a rejection at −0.0062** — keep the ReLU gate on the GNN residual
branch, do not re-run it. **`b3_wd_groups` is an exact cosine null (−0.0002)** that still narrows
both the drift and the train/val loss gap, so it is a hold, not a rejection.
`a6_ce_film` posts the best number (**+0.0063**) but carries the **−40.8% intensity-head cut** that
`attention_type: full` imposes, so it stayed unattributable until `a2_full_attention` ran —
which it since has; see the current handoff and §3 of the read-out. `a7_ce_both` (+0.0035) and `a3_full_massbias` (+0.0028) are
holds.

**Three things the read establishes beyond the gate.** (1) The 630-vs-128 horizon does **not**
confound the comparison: all eight runs have `lr_schedule: False` at a constant `lr: 1e-3`, so
nothing in the optimizer reads `max_epochs` and the horizon decided only when the job stopped. Two
further metadata traps are harmless but worth knowing — the `vulcan_l40s` tag on the six arms is
**wrong** (`run.metadata['gpu']` says all eight ran on Fir H100, within 35 minutes of each other),
and all six arms carry `c4a_base_s1`'s seed 397623, so each is a seed-matched paired comparison.
(2) The effects are **not additive**: `a7`, which stacks pre-LN onto CE-FiLM and both bias terms,
scores below `a1` and `a6` individually; and among the three arms that share the identical attention
change, CE-FiLM beats the mass pair bias by **+0.0035** at matched capacity and matched topology,
which makes CE the live half of review item 1. (3) **Coverage moves in none of them** —
`val_wrecall` spans 0.0006 across all eight runs while cosine spans 0.0125 — an eight-run
confirmation of the overfit read-out's allocation-not-coverage conclusion on a population it did
not cover.

**Next.** `b1_bond_sum` and `a2_full_attention` remain the lead pair, `a5_ce_pair_bias` gains
priority over `a4`, and a seed replicate of `a1_prenorm` is now worth a box: it is n=1 against an
n=2 control whose 0.0007 spread is tighter than this family's history, the same caveat the `d3`
LR-decay result carries, and `a1` and `d3` are the two live candidates that both act on
optimization under a constant LR and have never been run together. Validation: relative links
resolve, Ruff not applicable, no test run and none required for a documentation change. No run
launched.

Architecture-review todo consolidated (2026-09-22). **Documentation only; no code, config or
run changed.** The 2026-09-19 code-vs-literature review's remaining items are now one list:
[architecture review todo](plans/masster/masster_architecture_review_todo.md). Five of its seven items
were implemented 2026-09-20/21 and all default off. The blocking observation is that the six arms
that did run (`a1`, `a3`, `a6`, `a7`, `b2`, `b3`) were killed at the wall clock between epochs 179
and 210 and **have never been read at the 128-epoch screen length**, so four review items rest on a
W&B read that costs no GPU -- `a1` and `a6` lead the 25-run set at 0.7364 but over a longer horizon
than the 0.7282 control they would be promoted over. **That read was done later the same day; see
the handoff entry above for its verdicts.** Verified while writing it: every config in
`config/masster/ma_mi_merged/attention_2026_09_21/` still carries `min_epochs: 378 /
max_epochs: 630`, and resolved against the current template that schedule is their **only**
difference from `c4a_base_s1` besides their own keys, because `h_transfer_bound`,
`boundary_bond_encoder`, `candidate_mass_features` and `candidate_charge_feature` are template
defaults now; that schedule is why the ran arms were killed mid-run, and the five unrun arms
(`b1_bond_sum`, `a2_full_attention`, `a4`, `a5`, `b4`) need it set to 128 before submission.
`b4_budget_rebalance` is **not** a one-key arm -- it moves `bond_num_layers` 3 -> 1 while widening
the intensity head and the latent set.

**Fixed and measured 2026-09-22, same request.** The five unrun arms are now `min_epochs`/
`max_epochs` 128; the six that already ran were left at 630 so their config still matches the run
in the index. Each unrun arm now resolves to exactly its own keys against `c4a_base_s1` (`a2`,
`b1` one key; `a5` three; `a4` four; `b4` three) and all five construct. Constructing every arm
then surfaced an **unstated capacity confound**: `attention_type: full` deletes the induced layer's
latent attention block, feed-forward and inducing tokens, so every arm that sets it also cuts the
intensity head -- the module the campaign identifies as the allocator -- by **43.6%** (`a3`,
8,145,026 total) or **40.8%** (`a6`/`a7`). `a1`, `b1`, `b2` and `b3` are parameter-identical to the
9,215,573-param base and read directly; `a3`, `a6` and `a7` are *not* a mass/CE-bias test but
"bias + full attention + a 41-44% smaller head", so `a6` ties `a1` for the campaign's best number
(0.7364) while running 40.8% less allocator. None of the three is attributable until
`a2_full_attention` supplies the topology-only control, which promotes `a2` to co-lead with `b1`.
Same class of defect as the 2026-09-13 candidate-attention audit's capacity finding. RWSE is the
only item with nothing implemented and stays
blocked on porting the `random_walk_pe` `shape=` fix, which is on `feiw_v2` and
`project/spectraverse_benchmark` but **not on this branch**. The plan excludes the scheduler
forwarding repair, which is fixed in `031c370d`. Validation: relative links resolve, Ruff not
applicable, no test run and none required for a documentation change. No run launched.

MASSter candidate-token changes (2026-09-22). **LANDED in the working tree, both keys default
off, nothing trained.** Two one-key levers on the head that the
[overfit read-out](findings/masster/masster_overfit_readout_2026_09_22.md) blames for ~98% of the
0.151 cosine gap (allocation, not coverage: train `wrecall` 0.8975 vs val 0.8911).
**`masster_candidate_cut_bond_features`** pools `chemical_bond_attr[boundary_bond_indices]`
per candidate, summed and averaged, because `_boundary_embeddings` builds its block from atom
ENDPOINTS and never reads the bond attributes -- so a cut bond's own type/ring/conjugation reach
the intensity head only through endpoint mixing, which an internal atom of degree > 1 destroys.
The summed half is the multiset of broken bond types. Costs 2 x bond width (18 columns here,
1,285 -> 1,303) and no parameters. **`masster_candidate_atom_dropout`** drops each atom from
a candidate's membership set before the component pool, training only, attacking the identity that
pooling leaves behind (the cc atom mask resolves 1.00 distinct fragments per descriptor bucket).
Dropped atoms **migrate to the complement** rather than the complement being held exact, because
the complement is precursor minus candidate and an exact one would leak the true candidate sum by
subtraction; every other candidate input keeps the true membership, and a candidate is never
emptied. Arms, each verified a one-key diff from `c4a_base_s1` at the same seed and 128 epochs:
`config/masster/ma_mi_merged/candidate_token_2026_09_22/{e1_cut_bond_feats,e2_cc_atom_drop01,e3_cc_atom_drop02}.yml`.
**Correction to the [cut-set plan](plans/masster/masster_cut_set_features_plan.md):** its item C
(remove `use_base_probability_feature`) is not available as an arm -- `masster_pl.py:1008`
derives it from `masster_proposal_anchored`, which is already `true` in `c4a_base_s1`, so the
`c4a` token never carried `mask_log_probabilities`.

**Then the premise was remeasured on the right population, and it moved the design.**
[Candidate cut descriptors](findings/masster/masster_candidate_cut_descriptors_2026_09_22.md),
`scripts/debug/debug_candidate_cut_descriptors.py`, 400 molecules / 73,045 of MASSter's OWN
candidates rather than the depth-3 DAG the plan measured: **86.9%** are cut at >= 2 bonds (the DAG
said 96.8%) and the relational shares are about twice the DAG's -- **73.6%** of all candidates have
two cuts sharing a ring, **68.4%** two cuts sharing an atom. Within-molecule candidates per
descriptor bucket: cut count **47.5** -> pooled cut-bond sum **22.6** -> plus adjacency **10.4** ->
plus both ring terms **10.4**. Given the bond sum, the ring-sharing flag has **0.000** residual
ambiguity (the sum carries `b_ring` counts) while the adjacency flag has **0.489**. So the plan's
item A was redirected: adjacency is the missing relation, and it needs no SSSR ring identity --
which is not on the graph and would cost a new field plus a cache regeneration for 0.02-0.03 of
bucket resolution. Shipped as **`masster_candidate_cut_adjacency_features`**: two columns,
cut-bond pairs meeting at a shared atom, split by whether that atom is retained (one atom lost two
substituents) or lost (the fragment expelled a bridging atom -- the CO/CH2/S extrusion shape).
Arms `e4_cut_adjacency` and the stacked `e5_cut_bond_plus_adjacency` join e1-e3. **Counter-weight,
recorded in the finding:** the 2026-07-23 bond-set probe found the *least* identifiable fragment
descriptor generalized best, so resolving more candidates locates information the head cannot
otherwise get -- it does not predict the sign on validation cosine.

Validation: 19 new tests, the candidate-feature file 54 pass, **full suite 4,206 passed / 1
skipped / 2 xfailed**, Ruff and Black clean on every changed file, all five arms verified as
one-key (e5: two-key) diffs after template resolution. Smoke:
`config/debug/debug_tiny_masster_cut_features.yml` stacks all three keys at toy width with
dropout 0.3 and ran **green on `debug_tinyv4`**, 2 epochs, `detect_anomaly: true`, CUDA candidate
backend, no NaN, loss falling (train -0.509 -> -0.754, val -0.806 -> -0.981). Decomposed one key
at a time against the all-off control at three seeds: **every arm lands inside the control's own
seed band** (val cosine 0.196-0.225), the cut-bond block gives the best train loss of the seven
runs and adjacency the best val loss, and the only systematic effect is the dropout suppressing
the training fit. Added columns do not make the toy fit harder; the risk they carry is
generalization, not optimization. Parameter deltas land exactly on the token and are additive (+288 cut-bond, +32
adjacency, +320 both, +0 dropout at test width). No production run launched.

Feature and information audit (2026-09-22). **Documentation and diagnostics only; two of its
own conclusions withdrawn.** Synthesis with a per-claim status table:
[feature_information_audit_2026_09_22.md](findings/feature_information_audit_2026_09_22.md) --
read it before any of its five parts. Headline results: GLACIER publishes **no model size at all**
(ICEBERG 2.0 is 7.0M against FraGNNet 2.79M and MASSter 9.22M, all measured); descriptor
features are a **closed null** (MMFF/EState/Crippen/TPSA and the benzylic family are recoverable
from current inputs at 0.93-1.00); `b_stereo` cannot be matched without a preprocessing change
(0 of 127,249 bonds carry stereo); NIST ships an expert formula per peak covering **99.4% of
intensity** that never reaches the loss. In the tree, untrained: three new atom/bond features with
tests, `masster_candidate_composition_features` (off by default), three matched config arms,
and five diagnostic scripts. Proposed, not built: three MASSter candidate-token changes
([cut-set plan](plans/masster/masster_cut_set_features_plan.md)) -- **96.8%** of fragments are cut at
>=2 bonds and **55.4%** have two cuts sharing a ring, which mean pooling cannot distinguish from
unrelated cuts. **Retracted:** the proposal that MASSter needs a structural teacher; coverage
already transfers (train `wrecall` 0.8975 vs val 0.8911 against a 0.151 cosine gap). Validation:
full suite green, Ruff clean, links checked. No run launched.

Above-precursor target filter (2026-09-22). **LANDED in the working tree.** `spec_params` gains
`filter_above_precursor_peaks` / `above_precursor_ppm` /
`above_precursor_use_instrument_tolerance`, default off, applied in `clean_spectra` after
centroiding and before subformula filtering, with the same per-instrument tolerance and the same
retention statistics as its two sibling cleanups, plus the standalone CLI flags. The cutoff is
`prec_mz` widened by one matching tolerance -- deliberately the same rule
`scripts/eval/dag_oracle_scores.py` uses for its `1_above_prec` tier, which was the default for
every published oracle number while the training target kept those peaks. **Turning it on also
removes the precursor isotopologue ladder**, which is 72% of the affected intensity and is real
analyte signal; that is intended while isotopes are unmodelled (`num_isotopes` is 1 everywhere,
every DAG `isoFalse`), but it changes the target, so **cosines are not comparable across this
flag**. Validation: full suite 4,167 passed / 1 skipped / 2 xfailed; Ruff clean on every changed
file; 9 new tests; end-to-end on the real merged MA/MI val split the above-precursor intensity
goes **0.00899 -> 0.00000** and 10,339 of 792,217 peaks are removed, matching the 0.0090 predicted
by the split probe. Three existing `spec_params` fixtures needed the new required keys, which is
the intended strict-required behaviour. No run launched.

Intensity escape / dump (2026-09-22), same request. **LANDED END TO END** behind
`masster_intensity_escape`, default off, **never trained**.
`sparse_peak_list_distance` takes `pred_dump_intensities` (shape `(batch_size,)`, same units as
`pred_intensities`); when given it collapses every *unreachable* observed peak into one appended
coordinate and lets the model predict that mass, with all four objectives and the entropy term
running unchanged on the extended vectors. Motivation is the user's: `wrecall` is 0.891, so 10.9%
of observed intensity sits where no candidate reaches and the predicted distribution -- normalized
over candidates alone -- is forced to over-allocate everything it can reach. Worked exactly for
Hellinger, the escape removes an irreducible per-spectrum loss floor of `1 - sqrt(t(S))` that
varies with coverage and therefore weights training by how well enumeration did on that molecule;
**it does not change the optimal relative allocation**, which is `q_i ∝ t_i` either way, so it is
not a ranking change. It also makes `kl_distance` finite on a coverage gap rather than leaning on
`kl_epsilon`. Reachability is read from the tolerance `match_mask`, never from predicted values,
so a candidate the model scored zero -- or one a Hungarian assignment gave to a neighbour -- still
faces the loss. **This dissolved the plan's open "where is the target computed" decision**: it is
batch-time machinery that already exists, with no `spec_df` column and no `tolerance_consensus`
merge interaction. Validation: full suite **4,177 passed / 1 skipped / 2 xfailed**, Ruff clean,
10 new tests covering the algebra, the structural reachability rule, gradient flow, KL finiteness
and input validation; `pred_dump_intensities=None` is bit-identical to the previous behaviour on
all four objectives, under test. The model side is wired: `MassterModel` takes
`intensity_escape` and drives the intensity head's existing `predict_oos`, hard-coded `False`
until now; the head emits one logit per spectrum from its mean-pooled molecule token and the
model returns `pred_escape_intensities`. Parameterization: the escape's share of `candidates + escape`
is `sigmoid(logit)`, whose odds are just `exp(logit)`, so the emitted mass is
`candidate_mass * exp(logit)` with the logit clamped at 9. A code review caught the first
implementation going through `sigmoid` and back, which **broke under reduced precision**: fp16
and bf16 round any `1 - 1e-4` guard to exactly `1.0`, so the share saturated and the odds became
`inf`, crashing the step on the loss's finiteness check. The head's bias starts at `logit(0.1)`,
so training begins with the escape holding 10% of the spectrum, close to the measured 0.109, and
a test asserts it. The escape head also warm-starts from a checkpoint that predates it, matching
what `ion_h_support_head` already did, so switching it on does not force a run from scratch. `MassterPL` **raises at
setup** when the escape is requested with `masster_intensity_output: probability` or a
cross-entropy `loss_type` rather than ignoring it -- silent override is how `predict_oos` went
unnoticed on this model for months. The escape is reachable only in unique-fragment,
proposal-selection-only mode, the only mode `independent` intensity output supports. Also fixed: the
head's molecule pooling divided by a component count that can be zero, on a path that had never
executed because `predict_oos` was hard-coded off at its only call site. Validation: full suite
**4,184 passed / 1 skipped / 2 xfailed**, Ruff clean, 17 new tests in total for this item,
including an end-to-end one that runs the real model, feeds its output through `loss_fn`, and
shows the escape both reaches `sparse_peak_list_distance` and lowers the loss. **Smoke-trained green 2026-09-22**, not measured:
`config/debug/debug_tiny_masster_intensity_escape.yml` and a matched control differing only
in the flag, 64 toy spectra, 8 epochs, `detect_anomaly: true`, both exit 0. The head starts at
0.1 by construction and reaches `escape_share` **0.591** against a true unreachable fraction
`1 - wrecall` of **0.528**, so it tracks the coverage deficit; `wrecall`/`recall`/`precision` are
identical to the last digit between arms, the control the design requires; Hellinger halves
(0.746 -> 0.362) against a predicted floor of `1 - sqrt(0.4717)` = 0.313. **The loss drop is not
a quality gain** -- the objective now includes a coordinate the model can trivially get right, so
a lower number is guaranteed, and **cosine, the one metric whose definition did not change, went
slightly DOWN (0.2236 -> 0.2022)**. That is noise at this scale but is what the algebra predicts:
the escape is not a ranking change. `escape_share` is logged per spectrum whenever the escape is
on, since otherwise the head is invisible. **Real comparison must be on cosine at matched epoch
with replicates, never on the loss.** **Merged arms prepared, not submitted:**
`config/masster/ma_mi_merged/intensity_escape_2026_09_22/` is a **one-knob ladder** off the
loss-objective arm `h1_hellinger_closest.yml` (`oykkffls`, 0.7247), needed because `h1` keeps
`intensity_output: probability` while the escape requires `independent` -- so `e0` (independent,
escape off, 2 seeds) isolates the output mode against `h1` and `e1` (escape on, 2 seeds) isolates
the escape against `e0`. Verified by resolving each config and constructing the model: the
resolved diff is exactly the intended key at each step, and the escape adds **257 parameters** to
a 9,217,181-parameter model with no change to the memory shape, so the lineage's
`fir:h100-2g.20gb` profile still applies. Generation:
`scripts/manual_slurm/fir/submit_masster_intensity_escape.sh [ladder|control]` (generates
without submitting, validates configs and wheels first, syntax-checked). Nothing generated into
`slurm_scripts/`, nothing submitted. The
contaminant escape (`predict_oos` supervised by the subformula test, addressable population
0.0156) is still unimplemented; see [the plan](plans/masster/masster_supervised_escape_plan.md).
Isotope detection is **deferred by user decision** (2026-09-22).

Unexplained-intensity split (2026-09-22). Answering "what escape do we need for the peaks we
cannot explain" by measuring the population first; read
[the split](findings/masster/unexplained_peak_split_2026_09_22.md). Over the full 15,257-spectrum
`nist20_ma_mi_inchikey` val split, built through the run's own `spec_params` so the probed target
is the trained target, the 10.89-point residual behind `val_wrecall` 0.8911 divides **73%
coverage / 27% escape**: 2.91 points carry no precursor-bounded, nonnegative-RDBE subformula and
therefore cannot come from the molecule, while **7.98 points are real analyte fragments the
candidate enumeration misses**. The test is not vacuous -- a matched-mass donor formula rejects
0.5993 where the true formula rejects 0.0291, a **20.6x** discrimination. The escape population
is a diffuse small-peak background (an infeasible peak carries 0.21x a feasible one's intensity,
half the mass is spread over 7.9% of spectra, 23.3% of spectra are entirely clean) with a ~1.6%
tail where a dominant peak is non-analyte, **so a per-spectrum scalar escape is the wrong shape
for it** -- which agrees independently with the read-out's proof that a scalar `predict_oos`
cannot move a scale-invariant metric. Flat across adduct, instrument and fragmentation mode.
**46.5% of the escape population is isotopologues, not contaminants** (0.0135: the precursor M+1..3
ladder plus fragment isotopologues one or two neutron masses above an explainable peak), leaving a
genuine noise population of **0.0156** -- 14.3% of the residual. That half wants isotope handling,
which is scoped but never switched on (`remove_isotope_peaks` and friends are marked NOT HOOKED UP
in `template.yml`; `num_isotopes` is 1 everywhere and every DAG is `isoFalse`), so the model is
graded on peaks it structurally cannot produce. **Nothing removes peaks above the precursor**
anywhere in the training or validation path, while every oracle script (`dag_oracle_scores.py`,
`dag_oracle_decomposition.py`, `dominant_peak_diagnostic.py`) defaults to the `1_above_prec` tier
that does -- so published oracle ceilings were measured on a target the model does not train on.
A blanket above-precursor cut is nonetheless wrong: it removes 0.0090 of which 72% is the precursor
isotopologue ladder, real signal; only 0.0025 is genuine co-isolation. **Priority is coverage, not
escape** -- 7.98 points versus a 1.56-point contaminant population. Relevant existing machinery,
all currently off for MASSter: the `predict_oos` head is implemented in
`bond_fragment_heads.py:678` but hard-coded off at `masster_model.py:1508` and force-disabled
in `MassterPL`; `filter_subformula_peaks` is a hard per-peak version of this probe's test and
has never run on this base (0 of 58 runs). New probe:
`scripts/debug/debug_unexplained_peak_split.py`. Validation: Ruff clean; probe run end to end on
the full split plus a 400-spectrum smoke run; no model code changed, no training launched.

MASSter overfit read-out (2026-09-22). Read-only W&B review of the 25 runs of 2026-09-21/22;
read [the read-out](findings/masster/masster_overfit_readout_2026_09_22.md) before launching another
regularization, objective or schedule arm. The train/validation cosine gap reaches **0.151** by
epoch 127 and never plateaus, but it is **not a coverage gap**: train and validation `wrecall`
agree to 0.0064 against a 0.1507 cosine difference, worth ~0.0027 of achievable cosine at the
local `wrecall`->`opt_cos_sim` slope measured on three 2026-08-26 runs, so **~98% of the
generalization gap is intensity allocation among peaks the model already proposes**. FraGNNet on
this exact split logged the oracle on both sides and agrees (train `opt_cos_sim` 0.9271 /
validation 0.9243). Whether narrowing the gap helps is **untested, not refuted**: the
four-objective ordering is algebraically forced (val-on-train slope 0.605 makes gap and
validation rank together by construction), and in the 8-arm 09-21 regularization sweep
corr(gap, val) = +0.72 collapses to **-0.67** once the single arm `drop03` is removed. `wd10`
narrowed the gap without suppressing train cosine and moved validation **+0.0015** at 10%
exposure -- one full-length run would settle it. Validation *primary* loss turns before
validation cosine in all six instrumented runs but by **6 to 476 epochs** (the CE control leads
by 6, i.e. not at all); the 630-epoch control ends **0.096 worse** than its epoch-79 optimum,
non-monotonically, while its cosine moves **+0.0042** and never falls -- a calibration signal,
not evidence that late training costs cosine. The 128-epoch screen length is confirmed on 15 arms
(median +0.0031 from epoch 128 to best). **One live candidate:** the LR-decay arms, the first to
get a real schedule after the 2026-09-21 forwarding fix -- `d3` (cosine decay + 1000-step warmup)
reaches 0.7357 against a 0.7286 control mean whose replicates span 0.0007, but n=1; replicate
before promoting. **Two instrumentation gaps block the next read:**
`compute_train_auxiliary_metrics` is on in 4 of 25 runs and `opt_cos_sim` in none, and
`c4a_base_s1/s2` carry a `train_aux_metrics` tag while their resolved config says `False`. Also
pinned: the 0.73-vs-0.80 difference against `8v0i0fq7` is **scope plus the subformula target
filter**, not a regression -- a 256-run cross-tab puts ~0.06 on adduct/instrument/fragmentation
scope (4 adducts / FT+QTOF / HCD+CID vs `[M+H]+` / FT) and ~0.006 on `filter_subformula_peaks`,
which is `True` in `8v0i0fq7` and `False` in **all 58** `nist20_ma_mi_inchikey` runs. The filter's
share is not a matched eval: `clean_spectra` runs at dataset load, so each arm is scored against
its own cleaned target. The read-out now also records the intensity-memorization threat model:
candidate-set fingerprinting, unequal molecule exposure, CE conditioning proxies, population
priors, and non-causal fragment proxies are plausible but not yet causally isolated. The current
lead already excludes direct proposal probabilities, and unmerged runs also overfit, so neither
removing that feature again nor simply unmerging is a supported fix. Next gate: eval-mode,
molecule-balanced train/validation metrics with seen/unseen-scaffold slices, followed by matched
`wd10`, instrumented `d3`, intensity-only capacity/dropout, molecule balancing, and QTOF CE arms.
Documentation only; no code, config or run changed, and no training or evaluation was launched.

ICEBERG/GLACIER parity (2026-09-22). Closing the gaps behind "do they have a larger model, more
features?". **Model size:** GLACIER publishes none — no hidden dim, layer count or parameter
count, only `4xNVIDIA A5000`; its own baseline ICEBERG 2.0 is 7.0M on NIST'20 (measured from the
upstream configs), against FraGNNet 2.79M and MASSter 9.22M (both measured). **Features:**
ICEBERG carries strictly fewer than we do; GLACIER carried three we lacked, now added to the
shared registry — `a_fused_ring`, `b_ring_size`, `b_fused_ring` — plus the candidate token's
complement composition, `masster_candidate_composition_features` (GLACIER eq. 14's
`psi(r_form - f_k)`), off by default. `b_stereo` cannot be matched by a feature: preprocessing
strips stereochemistry, so 0 of 127,249 sampled NIST20 bonds carry any, on NIST23 and
MassSpecGym alike. Three matched arms are prepared and **none has been run**:
`c4a_ring_topology`, `c4a_complement_composition`, and FraGNNet `graph_feature_parity`.
*(Corrected 2026-09-27: all three ran on 2026-09-23. `3gqgkl4x` −0.0027 and `lxf97tck` +0.0012
are null; `m8qgi0ft` crashed at epoch 77 and has not been read.)*
Details in [the GLACIER note](notes/glacier_object_detection.md); validation: full suite green,
Ruff clean on every changed file; no evaluation, no cluster job. Two debug configs stack **all
four** new features at toy width and both smoke-trained green on 2026-09-22 (2 and 5 epochs on
`debug_tinyv4`, loss finite and falling, `detect_anomaly: true`, CUDA candidate backend):
`config/debug/debug_tiny_masster_glacier_feats.yml` and
`config/debug/debug_tiny_fraggnn_graph_feature_parity.yml`. Smoke runs, not measurements.

The shared agent workflow is established: start at [AGENTS.md](../AGENTS.md), follow the
[session workflow](workflow/session.md), and use the [architecture map](architecture.md).
Existing AGENTS and Claude rules are organized into [engineering](rules/engineering.md),
[Python and ML](rules/python.md), and [documentation](rules/documentation.md) rules.
The agent entry point selects the applicable files by task; the original rules are preserved.
This includes import aliases, formatting/docstrings, testing conventions, and documentation
placement. Where legacy Claude constraints differed, the original AGENTS policy is retained:
preprocessing changes require explicit authorization and compatibility work; PR validation runs
when practical, with omitted checks reported.
The adoption changes remain uncommitted for user review.

Validation: shared-rule preservation, relative documentation links, and Git whitespace checks
passed. The configured interpreter provides pytest and Ruff. No model code changed; no training,
scientific evaluation, or full test suite was run for this documentation change.

MASSter CMF alignment (2026-09-14): MASSter now emits the full configured H-transfer
ladder for chemically valid, non-degenerate CMF adducts, matching FraGNNet. It preserves
reachable CRF/CMF duplicate suppression and in-source-loss exclusion; the pre-batch support
filter shares the same contract. Existing checkpoints are diagnostic-only because their newly
valid CMF-H channels were not trained. Validation: 3,032 passed, 21 skipped, 2 xfailed; the
CMF plan records the required oracle-coverage rebaseline and matched-control training.

## Research navigation

- [Annotation-aware retrieval](plans/annotation_aware_retrieval_plan.md) is proposed
  (2026-09-25), documentation only. It asks whether support, the low-intensity tail and fragment
  annotations rank MassSpecGym candidates better than cosine. Step 1 runs on `ll04i00m` over the
  9,734 `[M+H]+` val queries with a CE (the `ms_gym_benchmark_1_5` val split), needs no annotation
  columns, and gates the rest. No MSGYM retrieval scorer exists yet. Next: write it and run cosine
  on a 200-query sample.
- [Energy-based spectrum reranking](plans/energy_based_spectrum_reranking_plan.md) is a proposed
  shared design of record with a FraGNNet anchor (2026-09-13). Next: reconcile the prior correction
  null, verify export with `--min_intensity -1`, and require oracle excess over matched random pools
  before implementing/training an energy head. Documentation only; no runs or implementation.
- [MASSter improvement plan](plans/masster/masster_improvement_plan.md) records the proposed
  unmerged sequence: corrected CE control, proposal-encoder comparison, Gaussian support-gated
  JSS, conservative output pruning, and measured throughput improvements. Five merged warmup
  configs are prepared for a CE/JSS/KL loss and intensity-transform screen. Next: verify checkpoint
  persistence before launching; resolve the unmerged anchor/warmup policy before its matched
  control. The plan includes [Vulcan generation and submission commands](plans/masster/masster_improvement_plan.md#vulcan-commands-for-the-five-config-screen)
  for all five configs with a three-hour allocation. No runs were launched for this plan.
- [MASSter code audit](findings/masster/masster_code_audit.md) records three reproduced
  issues at `0b5ad3f4`: standalone SMARTS inference, charged-component masses, and
  SMARTS proposal gradient isolation. All three fixes and regression tests are present
  at `77fdbea2`. A continued audit reproduced two additional inference blockers:
  unconditional DAG-file filtering/dataset selection in the CLI and independent-output
  tabular export expecting `pred_logprobs`. Those fixes and the mass-feasibility
  alignment are now in the working tree, along with CLI directory/row-layout fixes.
  Validation: 781 passed, 8 skipped; six CLI regressions passed after adding Parquet
  serializer-boundary coverage. Real Parquet serialization was not verified because
  its optional engine is unavailable; checkpoint loading and dataset preprocessing
  are mocked in the CLI tests. Next check: a target-environment real-checkpoint CLI
  smoke run. Review the linked compatibility
  notes and re-evaluate affected checkpoints before comparing experiment metrics.
  A third audit pass found two remaining API issues: prediction-only
  `run_inference_pkl` rejects its existing `pred_ints` output, and direct CPU bfloat16
  calls with multiple proposal mixtures fail the normalization check. Standard CPU
  runner fallback avoids the latter. The pickle API fix is now in the working tree:
  90 tests passed, including real-model prediction-only regressions for both output
  modes and batch offsets. The bfloat16 issue is deferred by user because it is not
  used in the current workflow. Separate failure reproductions remain in the audit.
  The checkout was clean at audit start, so the
  uncommitted-work description in the workflow-adoption handoff above is historical.

- [MASSter regularization sweep](plans/masster/masster_regularization_sweep_plan.md) is **EXECUTED with
  no promotions**; read the [campaign read-out](findings/masster/masster_campaign_readout_2026_09_21.md)
  (2026-09-21) before acting on the plan. It was the execution of that audit's sweep half:
  eight one-hour `fir:h100-3g.40gb` jobs on the
  merged MA/MI base `nist20_ma_mi_merged_fast_batch512_cuda_lb64`, comprising the six prepared
  one-knob arms (`layernorm`, `graphnorm`, `drop02`, `drop03`, `wd10`, `ctxnorm`) plus two controls,
  all with `compute_train_auxiliary_metrics: true` so a train cosine is logged for the first time in
  this model family. Four historical replicates of the base config set the floor at about 0.003 val
  cosine sd from epoch 20 on, so the screen resolves roughly 0.007 and promotes on mechanism plus
  margin; throughput moved 2.5x between two same-config runs on one day, so every comparison is at
  matched epoch, never at matched wall clock. The GPU profile is an open, measured question rather
  than a fixed choice: `loss_batch_size` is pure loss chunking (`loss.py:629-675`), so batch 512 fits
  a `2g.20gb` slice at `lb16` (~15.8 GiB projected against 26.9 GiB measured at `lb64`), but that
  slice carries 2/8 memory slices against the 3g profile's 4/8 for 57% of the price, on a workload
  whose NVML memory utilization runs 72-90%. Stage A therefore submits the control on both slices
  and lets the measured `min/epoch` ratio choose between 8 jobs at n=1 and 14 jobs at n=2. The plan
  excluded an LR arm because the `lr_schedule_type` forwarding was dropped in
  `SpectrumPL.configure_optimizers`; **that forwarding is fixed as of 2026-09-21**
  (`spectrum_pl.py:1782-1815`, `fragnnet_pl.py:1511`), so an LR arm is now available and the
  exclusion no longer applies. The plan records that the
  `cc_*` dropouts listed as unswept are FraGNNet-only and inert for MASSter. Configs resolved and
  one job script generated and syntax-checked locally outside the repository; nothing was generated
  in `slurm_scripts/` and no job was submitted at plan time.
  **Outcome (2026-09-19/20).** Six of eight arms produced data: `_drop02` was killed by the
  out-of-range predicted-peak bug about 70 minutes before its fix landed in `ffe5d53a`, and
  `_ctxnorm` was never submitted. A1-A5 pass — train cosine logged for the first time, `E* = 64`,
  controls agree to 0.0016, backend resolved `cuda` on all ten jobs. **A6 fails**: the 2g pilot
  OOM'd in `calculate_match_mzs` and the fan-out went ahead on `3g.40gb` three minutes later, so
  the slice ratio is still unmeasured. Read-out: 0 promote, 5 hold, 1 reject (`drop03`, −0.0149 vs
  the control mean). Only `wd10` (gap 0.122) and `drop03` (0.113) move the train/val gap off the
  control's 0.128; `layernorm` and `graphnorm` are exactly null. Every arm was still climbing at
  the one-hour wall, at 10% of its 630-epoch schedule, and the best validation epoch is the last
  epoch for all eight.
  **Horizon probe read out 2026-09-21 (`tkkesus9`, full 630 epochs, 8.97 h, 70 ep/h): val cosine
  saturates by ~epoch 100, not 630.** 0.7232@64 -> 0.7292@100 -> 0.7287@300 -> 0.7309@629, so the
  last 529 epochs buy **+0.0017**, inside the 0.0016 control replicate floor; the nominal peak
  0.7318@555 is noise. The sweep's +0.0034/10ep extrapolation decays immediately -- total headroom
  past ep64 was only +0.008. **The ep64 nulls therefore stand, and the correct screen length is
  ~128 epochs (~1.8 h), not 630 (~9 h)**; the saved wall clock should buy n>=2 replicates, since
  at a 0.003-0.005 replicate floor an n=1 arm cannot resolve what this family produces. The
  train/val gap **never plateaus** (0.55@64 -> 0.61@100 -> 0.81@300 -> 0.88@600) and validation
  loss actively degrades after ~ep100 (-2.526 -> -2.424) while train loss keeps falling: pure
  memorization, hidden by the best-val-cosine checkpoint. Caveat: the probe carries the
  2026-09-20 feature bundle, so its curve cannot be overlaid on the §1 arms to retro-size that
  screen -- the saturation shape is the transferable result, not the absolute level.
  **Proposal warmup is null-to-harmful on `ma_mi`** (`jvaj1lyw` vs `tkkesus9`: the only differing
  resolved-config key is `masster_proposal_only_until_epoch: 4`, same seed). Best-vs-best
  -0.0004, final -0.0021, and behind at every usable horizon: -0.0113@100, -0.0130@300,
  -0.0062@500, catching up only in the last ~80 epochs. The +0.0071 that motivated it was on the
  merged `nist20_inchikey` split and does not transfer; the four prepared `c4c_pwarm_*` arms are
  answered and should not run.
  **Re-read 2026-09-22 — that key does two things, and the second one is the story.** With
  `separate_proposal_encoder: False` (true for both runs),
  `set_proposal_input_gradients_enabled(proposal_only or proposal_encoder is None)` is `True` in
  **every** epoch, so the key also un-detaches the proposal branch for the whole run. `jvaj1lyw`
  is therefore the only `ma_mi_merged` run in which proposal supervision reaches the shared
  encoder, and a 4-epoch freeze cannot explain a −0.0113 deficit at epoch 100 or −0.0132 at 400
  in a run that saturates by epoch 100. **The persistent component is the gradient connection**,
  so the detached proposal branch should be left alone. Corollary: the 38 configs with staging
  *and* a shared encoder were all silently gradient-connected and are not clean tests of their
  nominal variable. See
  [head-input asymmetry §7](findings/masster/masster_head_input_asymmetry_2026_09_22.md).
  Next: run the `c4a_*` bundle-attribution arms, then the now-unblocked LR-schedule arms
  (`c4b_*`), then refill the `_drop02` / `_ctxnorm` holes.
- [MASSter regularization audit](findings/masster/masster_regularization_audit_2026_09_19.md)
  (2026-09-19) establishes that regularization and optimization have effectively never been varied:
  across 226 `model_type = masster` W&B runs, `masster_head_dropout`, `weight_decay`, `lr`,
  `gradient_clip_val` and `train_sample_weight` are identical in all of them, and only two runs vary
  any regularization or optimization setting without also changing the encoder. Both predate the
  `tolerance_consensus` merged-target change, so neither is usable evidence. Five prepared ablation
  configs under `config/masster/ma_mi_merged/` have never been launched. Measured mechanisms:
  BatchNorm, not dropout, drives the divergence between the candidate set optimized in training and
  the one evaluated (Jaccard 0.317/0.611 at dropout 0 under `batch`, exactly 1.000 under `layer`,
  `graph` and `none`); normalizing a residual branch pins its magnitude for `batch`, `layer` and
  `graph` alike, so switching normalization does not restore a layer's ability to attenuate itself;
  and `SpectrumPL.configure_optimizers` dropped `lr_schedule_type`/`lr_total_steps`/`lr_min_ratio`,
  giving 3.774e-04 instead of the configured 1.000e-05 at step 100,000. **That forwarding is fixed
  as of 2026-09-21**; runs before that date trained on exponential decay whatever their config said.
  Next: measure the BatchNorm candidate-set gap on a trained checkpoint before any new training arm.
  It also records the 2026-09-19 decision to drop the atom decoder going forward: epoch-matched it
  costs no cosine (-0.0018, inside the replicate floor) while running 1.8x faster per epoch, and
  `config/template.yml` already defaults it to 0, so the recommended normalization arm
  needs no config change. The decoder code path remains for checkpoint compatibility.
  Probes: `scripts/debug/probe_masster_regularization.py`. Audit only; no training run,
  no model-code change.
- [Cross-adduct loss headroom](findings/cross_adduct_loss_headroom_2026_09_17.md) is a NEW, GREEN
  data-side gate (2026-09-17) for grouping MS/MS by adduct and adding a paired loss. 46.5% of
  NIST20 molecules carry at least two adducts; 78,938 groups match on instrument, fragmentation
  mode and NCE; the InChIKey split never separates a molecule's adducts, so groups are
  leak-free and usable in validation. `[M+H]+` and `[M-H]-` of the same molecule share about
  40% of intensity as neutral fragments at 29x over control, but a molecule-independent
  neutral-loss transform fitted on disjoint molecules closes only 9.3% of the gap to the 0.950
  replicate floor, so roughly 92% of the cross-adduct difference is molecule-specific and an
  adduct embedding cannot supply it for free. Chosen formulation: a delta loss on the shared
  fragment subset, keeping adduct-conditioned proposals. Blockers to implementation are known:
  `group_id` never spans adducts (0 of 76,164 groups) and the shared-proposal path asserts one
  precursor type per group. Next: run the trained-model half of the gate against a multi-adduct
  checkpoint; both checkpoints in this checkout are `[M+H]+` only. No implementation started.
- [MASSter feature set audit](reference/masster_feature_set_audit.md) (2026-09-20)
  audited every feature that reaches a MASSter parameter, over all `model_type: masster`
  configs, with column constancy measured on a 5,000-molecule NIST20 sample (120,641 atoms).
  It found one real defect: `AtomDegree.featurize` passes a list into `one_hot_encoding`, so 12 of
  `a_degree`'s 14 columns were always the unknown bucket and substitution degree reached the GNN
  only as two unnormalized integers; in total 34 of 70 atom-feature columns were constant. It also
  found the candidate intensity head received no mass, m/z or composition at all; that `a_charge`
  was enabled in FraGNNet's newest configs and in no MASSter config, although MASSter is the
  model with an explicit charge-affinity head; and that the boundary embedding was degenerate in
  the two ways FraGNNet diagnosed for `boundary_pair` and fixed with `CutBondEncoder`.
  **All of these are now fixed for MASSter and are the default** (audit §11): `a_degree`'s
  one-hot repaired **in place** at unchanged width (the interim `a_degree_onehot` featurizer was
  deleted; the repair is shared with FraGNNet, so every pre-2026-09-20 checkpoint of either model
  must be retrained), `masster_candidate_mass_features`,
  `masster_candidate_charge_feature`, `masster_boundary_bond_encoder`, `a_charge` added
  to all 177 MASSter configs, the constant unique-mode token columns dropped, the unread
  required `atom_formal_charges` check removed, and the featurizer no longer mutating
  `CANONICAL_ELEMENT_ORDER`. Constant atom columns fall from 34/70 to 30/71. FraGNNet is
  deliberately untouched: `a_degree` still exists and no FraGNNet config changed, though the same
  defect applies to all 323 of them. **Every MASSter checkpoint written before 2026-09-20 is
  incompatible** (atom width 70 -> 71 and a changed intensity-head input width), and molecular
  preprocessing caches must be regenerated for `atom_gasteiger_charges`. Validation: full suite
  3,890 passed, 1 skipped, 2 xfailed; 29 new tests; the lead merged config resolves and runs a
  finite forward pass. `a_radical` and `a_chiral` were deliberately left in place; the audit §11
  records why.
  **Measured 2026-09-21: the bundle is worth +0.0073 val cosine at matched epoch 64**, clearing
  both the 0.003 floor this entry asked for and the 0.0016 control replicate floor. The horizon
  probe `tkkesus9` (`..._lb64_long_control`, `nist20_ma_mi_inchikey`) is the first run carrying
  the new defaults: 0.7232@64 against the 2026-09-19 regularization-sweep control mean 0.7159
  (`ri3zm215` 0.7152 / `7cdsx970` 0.7167). A full resolved-config diff of `ri3zm215` -> `tkkesus9`
  leaves exactly six substantive keys, all from audit §11: `a_charge`,
  `masster_boundary_bond_encoder`, `masster_candidate_mass_features`,
  `masster_candidate_charge_feature`, `h_transfer_bound` none->valence (independently
  measured free at -0.0002, so ~no contribution), and the `a_degree` one-hot repair, which is
  **invisible in the config** -- same name and same 14-column width, but 12 columns were dead
  before `97c41abc`. (`daa36f63` "update features" is comment reflow only, no behaviour change.)
  **This is a bundle, not an attribution**, at n=1 against n=2 controls and riding ~2,400 lines of
  other `src/` churn between the two batches, so no single component is established. The
  `c4a_*` arms isolate it: full bundle + seed replicate, four leave-one-out arms
  (`no_mass_feats`, `no_charge_feat`, `no_bbe`, `no_acharge`) and an all-off `c4a_bundle_off`
  control. Corollary: **every null in the 2026-09-19 regularization sweep was measured on the old,
  partly-dead feature set**, so those knobs are re-openable in principle.
- **Config layout: one template for every model** (2026-09-20).
  `config/template.yml` declares every model's keys - the 271 shared ones (paths, `spec_params`,
  `mol_params`, trainer, loss, logging, eval) plus all 258 model-owned ones - so every run
  resolves the same 529-key schema and shared code can read any key without first checking
  `model_type`. `model_type` selects code, not config: from the custom config, else an explicit
  `model_type` argument (a checkpoint's), else `frag_gnn`; `VALID_MODEL_TYPES` in `runner.py`
  only guards against a typo in the value. The file is the shared core followed by one
  clearly-marked block per model, with model keys flat and prefixed; `tests/test_config_templates.py`
  enforces that a prefixed key sits only in its own block and that the six unprefixed exceptions
  (`magma_params`, `magma_dp`, `ann_params`, `ann_fp`, `spectrum_encoder_params`,
  `retrieval_eval_size`) are declared by name.

  This **replaces the per-model split** attempted earlier the same day, which gave each
  `model_type` its own self-contained `config/<model>_template.yml`. The split's only benefits
  were trimmer hparams and foreign-key warnings; its cost was a new failure class - a key moved
  into one model's template while its reader stayed in shared code, breaking every other model at
  runtime rather than at validation time. That fired for `num_log_*` and the `masster_*`
  sampler keys, plus a cluster-staging variant (validation read sibling templates the job never
  staged) and a latent one where 8 of 10 templates declared `model_type: frag_gnn`, so a
  non-FraGNNet checkpoint merged with a silent custom config became `frag_gnn`. It also cost
  5,133 lines of YAML against 922, with 271 keys duplicated tenfold behind a drift guard that
  covered 28% of them. One schema removes the class: nothing can be present for one model and
  absent for another. The caller-supplied `model_type` now outranks the template default, which
  fixes the checkpoint case; a regression test pins it.

  Kept from the split, because neither depends on the file layout: **recursive validation** of
  custom configs (a typo inside `spec_params` used to be accepted silently) and the `model_type`
  known-set check. Dropped: the inert-versus-unknown distinction and the sibling-template glob -
  with one schema, a key the template does not declare is simply a typo, and a cluster job stages
  one file whose validation cannot differ from the submitting machine's.

  Historical configs were not rewritten: `config/` is tracked experiment provenance. Validation
  surfaces the same 18 configs that set keys no template has ever declared - 14 were already
  rejected before any of this work, 4 are nested typos that only recursive validation catches
  (`config/clip/nist20_spec_mol_{clip,siglip,soft_clip}_pretraining.yml` set undeclared
  `spectrum_encoder_params.*` sub-keys, `config/debug/debug_tiny_new_frag_mol_feats.yml` sets
  `frag_params.pyg_pe_embed_k`). None were edited. Nesting the metadata-conditioning keys into a
  `conditioning_params` section was tried and **reverted** - it invalidates every trained
  checkpoint for no behavioural gain - and the same argument rules out nesting model keys under
  per-model mappings. Five keys no code reads were deleted from the template and from the 203
  configs that set them: `wandb_root_dir`, `nb_formula_normalized_entropy_weight`,
  `group_sampler_type` (a leftover from the rename to `simple_group_sampler_type`, which is what
  `runner.py` validates), `debug_val_loop_trigger` and `debug_bypass_train_val`. The image-logging
  keys `num_log_train_images`/`num_log_val_images` were deleted with their code: both were 0 in
  every template, no config set them, and `SpectrumPL._log_images` was their only reader.
  Directories renamed on request: `config/models` -> `config/models_template` and
  `config/masster` -> `config/masster`. Validation: full suite green; `debug_tiny_merged`
  and `debug_tiny_masster` train end-to-end. Still open: which conditioning mechanism
  survives is the [conditioning refactor](plans/conditioning_refactor_plan.md)'s question, not
  this layout's.
- [Model improvement roadmap](plans/model_improvement_plan.md) remains the canonical source
  for research priorities and acceptance criteria. Its banner is dated 2026-07-24; reconcile
  its proposed next action with current code and artifacts before resuming it.
- [Documentation index](README.md) lists live, proposed, landed, and closed work, including
  MASSter plans. Those are recorded statuses, not verification of current execution.
- Read linked [findings](findings/) before reviving a rejected or closed experiment.

No training processes, scheduler jobs, checkpoints, or external run statuses were audited during
workflow adoption. No new research priority or scientific conclusion is inferred here.

## Next task

The [FastGNN proposal option](plans/masster/masster_fast_gnn_proposal.md) is implemented as an
explicit alternative to cut-mask proposals, preserving the default and checkpoint shapes.
Full suite: 3,451 passed, 96 skipped, 2 xfailed; three separate GPU checks and a real-data
forward/backward smoke passed. The linked plan supplies the config and score semantics.
The connected-fragment option now aligns search and scoring with boundary-cut/internal-intact
probabilities from the updated prototype; see the linked plan for changed checkpoint semantics.
Three merged MA/MI configs (128/256/512 fragments and matching beams) and a generation-only
Vulcan launcher with three-hour jobs are prepared in the linked plan.
The [connected-fragment audit](plans/masster/masster_fast_gnn_proposal.md#connected-fragment-audit-2026-09-17)
found two selection issues now fixed in the working tree: stable beam ties remove padding
dependence, and complements are scored before output-budget pruning. Permanent regressions
and model/backend checks: 220 passed, 11 skipped in the sandbox; the six proposal CUDA checks
subsequently passed outside it. The [real-graph speed audit](findings/masster/connected_fragment_beam_speed_2026_09_17.md)
measured candidate time 238.7 → 732.1 ms and forward/backward time 744.4 → 1,252.3 ms at batch
128 / beam 256 on RTX 2070 SUPER. Complement connectivity dominates; peak whole-step allocation
stays near 2.83 GB. Next: reduce unnecessary connectivity checks without changing selection,
then measure L40S throughput and coverage. No production code changed during the speed audit.
All four connected-fragment arms now use `masster_fast_gnn_max_boundary_cuts: 5`: at 3 the
boundary limit, not the fragment target, was the binding candidate budget, so the 256 and 512
arms filled only 63% and 40% of their budgets and ran at roughly half the `cut_masks` candidate
count. Template and checkpoint-migration defaults stay at 3.
On real NIST20 batches the beam is **not** a throughput improvement: 1,024 ms median against
70 ms for `cut_masks` with `candidate_backend: cuda` and 157 ms for its CPU backend, at batch
128 and comparable fragment counts. Batch cost is set by the batch maximum heavy-atom count,
whose median is 94 at batch 128, and 77% of the time is the per-growth-round complement
reachability loop introduced by the audit fix. See the
[real-precursor measurement](plans/masster/masster_fast_gnn_proposal.md#real-precursor-throughput-measurement-2026-09-17)
for the profile and the proposed restructuring. Next: screen this arm on accuracy only, or
restructure complement connectivity before making any speed claim; no training run was launched.

MASSter training speed (2026-09-17): the
[s3gy1ro4 audit](findings/masster/masster_training_speed_2026_09_17.md) identifies monitoring-only
channel attribution and base CE as removable overhead. A matched fast config and reusable
benchmark are prepared; batch-128 local GPU forward/backward throughput improved by 1.40×.
Scientific losses and checkpoint shapes are unchanged; the two diagnostic families are disabled
in validation too. Loss/gradient equivalence and checkpoint compatibility passed. The small
attention-layout experiment was removed after failing to demonstrate useful whole-batch gains.
A batch-256/accumulation-1 variant is also prepared, preserving effective batch 256. It fits the
local sampled GPU workload (7.10 GB) but shows no throughput gain over batch 128; see the same
finding for matched measurements. Next: compare 128×2 and 256×1 on L40S and verify resolved resume
flags; no remote run or cluster job was changed. The requested 512×1 config is also prepared
with LR unchanged; its effective batch doubles to 512 and its GPU validation remains pending.

MASSter proposal loss (2026-09-18): the per-spectrum proposal coverage loop is now a
batched whole-batch match, measured 118.6 ms to 2.4 ms per call locally at production shapes
with the previous loop kept as the test reference; see the same
[training-speed finding](findings/masster/masster_training_speed_2026_09_17.md). No end-to-end
training-step measurement was run.

The [candidate-backend plan](plans/masster/masster_candidate_backend_plan.md) is active. CPU numeric
construction now releases the GIL and shares a persistent thread pool; local full-training
throughput improved about 9.3% with exact candidate/gradient checks. The score-free selector
specialization was removed after failing its throughput gate. Membership compaction and streaming
are deferred by the current projections. The same [speed finding](findings/masster/masster_training_speed_2026_09_17.md#cpu-implementation-and-full-training-measurements-2026-09-17)
records results and limits. Batch256 fits locally; batch512 runs out of memory on the 8 GB GPU.
[Three-hour L40S benchmark commands](guides/benchmark_masster_cpu.md) are prepared; no jobs were
submitted. CPU remains the default. The exact optional CUDA backend now passes local candidate,
model-gradient and checkpoint checks, with 19–23% higher full-training throughput than
improved CPU in the final 128/256 matrix. Explicit CUDA configs are available for all three batch sizes; see the same guide
for supported modes and compiler requirements. Final CPU full suite: 3,422 passed, 93 skipped,
2 xfailed; separate CUDA suite: 118 passed. Batch512 hits the local 8 GB GPU limit with either
backend. Next: confirm throughput, memory and native compilation on L40S before production promotion.

Merged MASSter arms for the candidate backend (2026-09-17): six configs under
`config/masster/ma_mi_merged/` give merged arms at batch 128x2 / 256x1 / 512x1 on both the CPU
and CUDA candidate backends. They are derived from the unmerged fast arms but are **not** a
controlled A/B on merging: by user decision `spec_params.ace` and `masster_include_cmf` are
both off, because no merged MASSter config has ever had a live CMF channel together with dual
CE (`context_ablation`/`smarts` set `ace: True` but sit on the `[M+H]+`/FT-only `nist20_inchikey`
split, so their Na/Cl adducts are absent from the data; the one merged config with live CMF is
NCE-only). Merging is therefore confounded with CE conditioning and the CMF channel; isolating
merging needs unmerged siblings with the same two flags off, which do not exist yet. Epochs are
378/630 rather than 60/100 because merging by acquisition `group_id` collapses train 287,721 ->
45,639 rows and val 96,153 -> 15,257 (6.30x), so the row budget matches (28.75M against 28.77M).
Merging does not strip adducts: 5,993 `[M+Na]+` and 748 `[M+Cl]-` of 45,639 merged train rows, no
group mixing adducts, instruments or frag modes. A merged row conditions on the mean-pooled
acquisition CE ladder (median 4 readings, max 86), not a single energy. Validation: all six
resolve through `load_config` with exactly the intended delta; the merged dataset constructs on
real NIST20 data; `tolerance_consensus` merging of whole multi-spectrum groups produces consensus
peaks; the CUDA config builds a model resolving `candidate_backend` to `cuda` (8,427,349 trainable
parameters against 8,434,087 with ACE/CMF on); 186 config/merge/fast-GNN tests passed. Merged and
unmerged validation cosine score different targets and are not comparable. Not measured: any
training step, batch-512 memory, epoch wall time, and the 6.3x larger fixed per-epoch overhead of
a 630-epoch run. See the [merged arms guide](guides/benchmark_masster_merged_control.md).
One-hour L40S jobs are the right size for the backend comparison but not for training: merged
rows fall 6.30x but peaks per row rise 31.9 -> 51.9, so total peak work falls only 3.83x, putting
a merged epoch at roughly 5.3-6.5 min at batch 128 against the measured unmerged 33.52 min
(estimate from row/peak counts, not a measured merged epoch). One hour yields about 8-10 epochs,
so 6 one-hour jobs cover the CPU-vs-CUDA throughput/memory comparison, while 630 epochs would
need 63-79 jobs per arm chained through `--dependency=afterany` with `j=` resume. The guide
carries the arithmetic. No job was generated or submitted.

Live fast-run follow-up (2026-09-17): the existing L40S runs `dny62x0i`, `h0ibnndb`, and
`du6cr9dx` still report the earlier separate candidate timing fields and no backend setting.
Four CPU selection threads are enabled, but the new parallel topology/CUDA implementation is
not reflected in those runs. Monitoring-only fast128 reduces epoch time from 44.5 to 33.5 minutes;
256/512 take 32.1/31.8 minutes, so larger batches alone leave the CPU bottleneck. The old CPU
batch512 run does fit L40S (about 14.76 GB median sampled allocation); new-backend L40S acceptance
remains open. See the [live follow-up](findings/masster/masster_training_speed_2026_09_17.md#live-fast-run-follow-up-earlier-candidate-code-still-running-2026-09-17).
Next: launch a fresh job from transferred updated source; running job-local installs will not
pick up these working-tree changes. No remote jobs were modified by this read-only audit.

Unmerged generalization audit (2026-09-16): the
[27-run W&B census](findings/masster/masster_unmerged_overfitting_2026_09_16.md) supports
widening train/validation loss gaps in mature runs, with modest continued cosine gains.
The current `s3gy1ro4` run was still improving at epoch 8. No train cosine is logged.
This audit corrects the anchor provenance below: `ef7fl5c6` is **merged**, not unmerged.
The claimed architecture-invariant 0.688–0.693 band excludes many of the 27 runs.
Current `GroupSampler` does resample through the epoch hook; MASSter's inherited
optimizer setup ignores cosine scheduler options. Next: matched evaluation-mode train/val
checkpoint metrics, then separate fixed-objective LR and regularization controls.
Documentation and a sampler probe only; no training or model/config changes.

Unmerged MASSter CMF anchor prepared (2026-09-16): the
[improvement plan](plans/masster/masster_improvement_plan.md#corrected-unmerged-cmf-anchor-prepared-2026-09-16)
records a new config,
`config/masster/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_unique256_enum4096_gine_gsa384_bondcentric_a2_b3_transformer_full_context_dual_ce_proposal_selection_only_inst_tol.yml`,
and a generation-only launcher
`scripts/manual_slurm/vulcan/submit_masster_unmerged_cmf.sh`. It is the lead unmerged arm
(`ef7fl5c6`, 0.7515) plus dual CE, `inst_types` reduced to the FT/QTOF actually present in the
split, channel diagnostics, and `ace_max: 300` (the template's 100 clamps 3.49% of this split's
spectra into one saturated embedding row once ACE is on); a resolved-config comparison shows
`spec_params` and `ace_max` as the only differing keys. Two conditioning differences against the
FraGNNet MA/MI reference are left in place as unmeasured treatments rather than defects:
`masster_context_in_gnn` and `masster_smarts_templates`, both `False`. The prepared
merged `context_ablation` pair is the place to settle the first. A two-arm proposal-warmup
pilot (`_warmup_e0_{2,4}.yml` beside the anchor, launcher
`scripts/manual_slurm/vulcan/submit_masster_unmerged_cmf_warmup.sh`) supplies the bounded
matched pilot that step 3 of the improvement plan asks for; each arm differs from the anchor by
`masster_proposal_only_until_epoch` alone, and the anchor is the no-warmup control. A real
four-epoch smoke of `e0_2` crossed the boundary with validation cosine flat through the frozen
epochs and rising once the intensity head unfroze. This is the first config to train the 2026-09-14 CMF alignment, because only its
`[M+Na]+`/`[M+Cl]-` spectra (38,156 and 7,820 of 480,002) have a non-degenerate CMF ladder. The
twelve-hour allocation and `j=` resume address the wall-time half of the convergence blocker that
holds up [Arm C](plans/masster/masster_ion_h_channel_selection_plan.md). `filter_subformula_peaks`
stays off so the target spectra do not change alongside the candidate universe. Validation:
config resolved, both arms constructed, generated job script passed `bash -n`, and one full
subsampled epoch ran end-to-end on the local GPU (exit 0, `val_cos_sim_0.01_epoch/mean` present,
`last.ckpt` written). No Vulcan job was generated or submitted from this workstation, which has
no `sbatch`. Next: run the launcher on the Vulcan login node, `sbatch` it, and record the
scheduler and W&B run IDs in the improvement plan.

Subformula filter hooked up to MASSter (2026-09-16): the merged Cl/Na fix reaches MASSter
through the shared `BaseDataset` cleanup with no code change, but it was **inert for every
MASSter config** — all of them use `data/split/nist20_inchikey`, which is 209,698 `[M+H]+`/FT
spectra with no Na, Cl, QTOF, or IT. Added
`config/masster/merged/subformula_filter/nist20_ma_mi_merged_independent_jss_warmup_e0_15_dense_subformula.yml`,
identical to its `[M+H]+` sibling except for `split_dp: data/split/nist20_ma_mi_inchikey`,
`prec_types` (four adducts) and `inst_types: [FT, QTOF]`, so the two remain a matched A/B on adduct
scope. MASSter needs no DAGs, so the split change costs no preprocessing. The
[subformula finding](findings/subformula_filter_ft_qtof.md#clna-budget-fix-measured-on-the-multi-adduct-split-2026-09-16)
records the measured pre-fix/post-fix delta (+90 peaks of 62,702, every adduct nonnegative),
per-adduct retention at instrument tolerance, and the model/filter agreement on CMF validity.
Validation: 344 passed, 2 xfailed across the filter, MASSter, dataset, and split-filter suites;
config resolved through `load_config`; cleanup run end-to-end on 2,620 real multi-adduct spectra
with zero emptied spectra. No training run was launched and no checkpoint was evaluated.
Next: launch the new config as the matched multi-adduct arm, or first measure oracle coverage on
the Na/Cl population. Note the existing 4-adduct MASSter configs (`context_ablation`, `smarts`)
declare Na/Cl in `prec_types` but sit on the `[M+H]+`-only split, so their extra adduct slots and
CMF channels have never been exercised.

Inference path consolidation merged from `feiw_v2` (2026-09-16): the
[inference step consolidation plan](plans/inference_step_consolidation_plan.md) is LANDED on that
branch and is now merged here. It collapses the three `SpectrumPL` inference entry points into one,
deletes `resolve_pred_ints` and its `step_untransformed` flag, and removes
`GreedyInferenceBatchSampler` in favour of the lazy `IterableSpecMolFragDataset`. The same merge
brings the subformula-filter fix for Cl and Na adducts, HDF5 output compression, and the
`drop_empty_spectra` prediction-only fix. Training and loss paths keep `pred_logprobs` and were not
touched. Two contract changes from `feiw_v2` win over this branch's prior behaviour: the exporters
now rescale each spectrum's base peak to 1000 on top of the L1-normalized `pred_ints`, and
`run_inference.py` no longer offers `--save_format parquet`. Two MASSter tests that pinned the
old exporter scale and the parquet writer were updated accordingly; this branch's `requires_dags`
guard and its `pred_intensities` output contract are preserved. Validation: 3,234 passed, 1 skipped,
2 xfailed; Ruff on the resolved files matches the `feiw_v2` baseline. Next: a real-checkpoint
inference smoke check; no GPU run or evaluation was performed for the merge.

Candidate attention topology audit (2026-09-13): the
[audit section](findings/masster/masster_code_audit.md#candidate-attention-topology-audit-2026-09-13)
records seven findings on the uncommitted `full` candidate-attention arm, all fixed in the
working tree: a hyperparameter fallback default plus a missing `get_pl_hparams` migration,
corrected A/B parameter counts, an unstated capacity confound, a missing unique-fragment guard,
a pre-existing CUDA bfloat16 `index_add` crash, two documented degenerate head inputs, and a
per-layer host sync in `_efa_linear_attn`. Bucketed attention masking was verified exact for both
topologies, and `full` is cheaper only up to about 256 candidates. Validation: full suite plus
274 focused tests, Ruff check and format on changed files, and construction of both A/B arms.
A pre-rebuild full-suite run gave 92 failed, 2,873 passed; every failure was the then-stale
multi-cut Cython extension, all carrying `compute_ccs_multi_cut()` unexpected-keyword errors
for `max_ring_cut_steps` or `ring_open_mask`. Both extensions were rebuilt in place at
2026-09-13 18:12, and `masster_core` was force-rebuilt at 22:38 to a byte-identical
binary, confirming the 18:12 build already matched its source. All four extensions are now
current and the full suite passes: 2,975 passed, 1 skipped, 2 xfailed.
Next: decide whether to remove the degenerate `component_residual_head` and constant mask-count
feature, which would invalidate existing MASSter checkpoints and so needs explicit
authorization. No training ran and no run was launched for this arm.

D4 oracle verification (2026-09-13): the
[subformula-filter finding](findings/subformula_filter_ft_qtof.md#saved-subformula-filtered-d3d4-comparison-verified-2026-09-13)
records reaggregation of the saved D3/D4 artifacts. MASSter's roughly 94% coverage is strong
relative to that unmatched D4 reference; the remaining 6% is not established cut-scoring headroom.
No preprocessing, DAG generation, or model inference reran. Next: compare current MASSter
support-oracle and predicted cosine on identical targets. Bond-cut-first remains the requested
diagnostic order, with training conditional on useful support or efficiency headroom.

MASSter H-support ablation (measurement 2026-09-14; code clarification 2026-09-15): the
[ion/H channel plan](plans/masster/masster_ion_h_channel_selection_plan.md#h-support-ablation-measurement-2026-09-14)
now records the matched 2026-09-08 gated / supervision-only / no-support arms read from W&B. No
trained arm exercises channel selection: retained channels equal valid channels in every arm
(about 8.5 per fragment against roughly 1.02 attributed), so the hard-selection stage has never
been measured. The soft support gate shows no spectral or weighted-precision benefit in either
data variant, and all dense cosine differences fall below one per-epoch standard deviation of a
single seed. The plan now records clone-initialized self-gating as an untested explanation for
the gate null, and clarifies that the six-run scope tests only H support (one valid ion mode,
at most nine CRF H channels). The support BCE does concentrate 95% of per-fragment channel mass into about 11%
fewer channels without moving cosine. Next: disable the gate in the H-support configs, verify the
`required_ion_h_top_k_95`/`_99` metric that reports identical means, and recover one cluster
checkpoint so the Phase 2 selection sweep can run; W&B holds no checkpoint artifacts for these
runs. Documentation only; no training, evaluation, or code change.

Joint selection objective (2026-09-13): the
[ion/H channel plan](plans/masster/masster_ion_h_channel_selection_plan.md#joint-fragmentchannel-selection-loss-experiment-2026-09-13)
now records a proposed single selection loss that couples fragment proposal and
ion/H support probabilities while retaining the independent spectral intensity loss.
The current JSS + proposal loss + support BCE configuration is the control. Next:
check joint-loss gradients and the all-channels-open failure mode on synthetic cases,
then compare on matched validation support and intensity metrics. Documentation only;
no implementation, config change, or training run.

MASSter SMARTS review (2026-09-13): the [set/literature review](findings/masster/masster_smarts_review_2026_09_13.md)
confirms 26 rules, 19 legacy rules without enforced ionization guards, and a pyrimidine
gap in the carbamate family. Positive-ion oxygen-transfer and prazole-sulfone candidates
are supported literature leads, not measured coverage gains. Validation: 75 SMARTS tests
passed on CPU plus registry/matching probes. Next: measure existing rules' incremental
coverage on filtered validation peaks before adding narrowly mapped rules. No model edits
or training ran.

MASSter proposal review (2026-09-13): the [proposal review](findings/masster/masster_proposal_review_2026_09_13.md)
prioritizes support diagnostics before encoder expansion and separates established loss/routing
limitations from exploration, diversity, and mixture hypotheses. A fresh 155-run W&B query
and eight full validation histories support selection-only in a close same-seed historical
pair: +1.54 percentage points candidate weighted recall and +0.01261 cosine at epoch 142.
A separate larger-intensity-head pair gains 0.00897 cosine at epoch 98 despite slightly
lower coverage. Recent matched JSS/KL results are close and do not establish a KL advantage.
Next for the intensity stage: preserve selection-only and diagnose fragment-versus-H ranking
on fixed support before another head or loss comparison; see the review for provenance limits.
A live W&B check of three
recent merged U256/E1024 histories confirms only 5.5–7.0% reach U256 at their latest logged
validation epochs. Two existing unmerged pairs match in logged configuration except E and
labels: E4096 adds about 43–46 fragments but only 0.66–0.67 percentage points of candidate
weighted recall at common epochs. A subformula-filter query found 14 runs, all U256/E1024;
late-run coverage is about 94%, with no filtered E4096 counterpart found. No new tests or
experiments ran. Next: inspect remaining
missed intensity and support-oracle headroom before proposing another budget run. The proposal review
is present but untracked, as is its linked loss review; include both in the eventual documentation
change set with these indexes. No files were staged or committed.

EBM modeling proposal (2026-09-13): the user selected both approaches, with
[bond-cut EBM modeling first](plans/masster/masster_bond_cut_ebm_plan.md), then spectrum reranking.
The [MASSter program](plans/masster/masster_energy_based_model_plan.md) records this ordering and
the spectrum setup delta. Next: select a checkpoint and validation subset, measure missed-intensity
and finite-mask-pool headroom, then evaluate pairwise cut energies against search-only, unary,
random, and mixture controls. Preserve probability contracts; the existing exact heap guarantee
does not extend to pairwise energies. No implementation or experiments started for this program.
The spectrum stage retains its separate shared protocol and null-adjusted gate.

Loss setup review revalidated (2026-09-21, `551a64ae`): the
[current assessment](findings/loss_setup_review_2026_09_13.md#current-assessment-tighter-objectives-2026-09-21)
reproduced all four FraGNNet defects on the starting source. Two vectorized-JSD defects are now
fixed; the current focused suites report 393 passed and 6 skipped. CPU probes reverified. A
separately named MASSter
`conditional_cross_entropy` path is now in the working tree while historical `cross_entropy`
remains unchanged; both explicit CE variants are logged. FraGNNet's vectorized Hungarian JSD now
uses the scalar JS-gain assignment and supports CPU fallback backward; regression probes agree
within 1e-6. MASSter direct losses and vectorized FraGNNet JSD share an explicit sparse-support
alignment primitive while retaining their different matching policies. No training was run.
Explicit zero-safe `hellinger_distance` is now wired into MASSter's closest/Hungarian direct-loss
path and FraGNNet's sparse binned path. Focused value and backward tests include exact structural
zeros and reproduce the suppressed-correct-peak probe gradient of `-0.004046`; no Hellinger training
run has been performed. The two auxiliaries are now implemented and default off:
`spectral_leak_loss_weight` penalizes the
normalized intensity outside every true tolerance window independently of Hungarian assignment,
and `spectral_ce_auxiliary_weight` applies historical coverage-weighted sparse Gaussian CE. For
independent MASSter output, candidate scores are L1-normalized before CE. Both terms are restricted
to Hellinger configurations and logged separately; their coefficient sweeps remain unrun.
Next: correct the formula-validity contract and settle OOS semantics, then measure
supported-peak ranking and per-term gradients before a targeted ranking experiment.
The recorded campaign does not establish a benefit from a generic JSD-to-KL replacement.

Second-pass loss audit (2026-09-21): the working-tree loss changes above were audited as changes
rather than against the older defects; see the
[second-pass section](findings/loss_setup_review_2026_09_13.md#second-pass-audit-of-the-working-tree-2026-09-21).
Four defects in them are fixed: the vectorized Hungarian JSD returned NaN for a spectrum with
observed but no predicted peaks (the emptiness guard is per chunk, and `551a64ae` scored the same
input 0.0, a perfect score for an empty prediction); the infeasible-pair sentinel and the SciPy
backend broke under float16/bfloat16 where a dropped `.float()` had been load-bearing; the leak
term inferred `batch_size` from observed peaks, so a spectrum without observed peaks shortened the
result and broadcast silently onto a per-spectrum loss; and the leak term ran an unbatched
per-spectrum loop with three to four device synchronizations per spectrum. The leak term is now
the chunked padded form (8.9x faster at batch 512 on CPU, values identical) and takes an explicit
`batch_size`. **Compatibility: `jss_hun`, `jss_hun_sqrt`, `jss_hun_np`, and `jss_hun_inst_tol` move
with the gain-matching change** — 40% of 300 random padded spectra change, mean +0.0165, maximum
+0.2653, never worse — so those metrics are not comparable across it. `cos_hun` is unaffected.
Cleared on measurement: the shared alignment primitive is bit-identical to the code it replaced
(`max |new - old| = 0` over 180 spectra, three objectives, both policies), vectorized and scalar
JSD agree to 6.3e-07, sparse Hellinger matches a dense reference to 1.8e-07, and conditional CE
agrees across the sequential and vectorized CE paths. `1 - BC` is the squared Hellinger distance
`H^2`, recorded in the docstrings and config comment. Validation: full suite **4068 passed,
1 skipped, 2 xfailed**; Ruff and `git diff --check` clean; CUDA assignment unavailable, so the
CUDA backend and any training remain unrun.

Both `hellinger_distance` paths were checked for batching and are batched: MASSter's objective
reduces inside the `loss_batch_size` chunk loop of `sparse_peak_list_distance`, and FraGNNet's
binned `sparse_hellinger_distance` has no Python loop. The gap was coverage: the objective had
been omitted from the four parameterizations in `tests/test_masster_peak_list_distance.py`,
including the per-spectrum loop-reference harness, so its chunked reduction had never been
compared against a reference. The reference now has an independently written Hellinger branch,
the objective is parameterized everywhere the others are (+57 cases), and chunk invariance,
batch-composition independence, per-spectrum gradient isolation, and a closed-form shared-bin
case were added for both paths. Three mutations confirm the coverage bites.

Original loss setup review (2026-09-13): the [loss review](findings/loss_setup_review_2026_09_13.md)
reproduces new FraGNNet vectorized-JSS correspondence/backward defects, ignored OOS
supervision under JSS, and invalid formula slots stealing support targets. MASSter
probes identify weak JSS recovery gradients, correspondence/support ambiguity, proposal-loss
coverage limits, and a no-warmup gradient-routing confound. No implementation changes or
training ran; 195 focused tests passed on CPU. Next: fix the verified FraGNNet contracts,
then inspect MASSter gradients/coverage before an isolated matched JSS/KL comparison.
The prior audit's three FraGNNet loss/explanation fixes and MASSter precursor-pair
rejection are already present in the reviewed source; their earlier pending status below
is historical. The W&B screen report is missing from this checkout; its historical summary below
is not reverified evidence.

Correctness audit (2026-09-12): the [combined model audit](findings/fragnnet_masster_audit_2026_09_12.md)
records reproduced checkpoint-loading, HDF5 cutoff, FraGNNet loss/explanation, and shared
test-metric defects, plus silently ignored MASSter precursor-pair supervision.
No fixes were applied. The existing local multi-cut extension is
stale; an isolated rebuild from current source yielded 2,876 passed, 21 skipped, and
2 expected failures for the full suite. Next: address checkpoint loading and test-metric
initialization, then the remaining counterexamples. Pre-existing SMARTS warmup edits
were preserved; CUDA and real-data scientific behavior remain unverified.

W&B review (2026-09-11; historical summary, source report unavailable here): the recorded screen
superseded older “no runs launched” statements. `docs/findings/wandb_screen_2026_09_11.md`
is absent from this checkout; recover the report before using these comparisons as evidence.
JSS warmup 0–4 leads the single-seed filtered screen at 0.80099 best validation binned
cosine; CE remains close and finished sooner. Support supervision helps early training;
an additional soft-gate gain is unresolved. GNN context completed, but its matched
heads-only comparison is missing from the inspected runs. SMARTS warmup failed at the
known construction guard. Next: matched-seed confirmation and the missing controls;
recover the missing report for metric definitions, run links, and provenance limitations.

SMARTS merged readiness (2026-09-09): the [readiness audit](findings/masster/masster_code_audit.md#smarts-merged-run-readiness-2026-09-09)
passed 494 CPU tests (5 skipped) and a synthetic SMARTS + merged NCE + GNN-context
forward/backward probe. The current merged configs are blocked by their proposal-only
warmup, which explicitly rejects SMARTS. The requested no-warmup SMARTS config is now
prepared and linked in the audit; full model construction and intended-only resolved config
differences passed. Next: perform a real-data GPU smoke check. No runs launched.
The subsequently requested [SMARTS warmup config](../config/masster/merged/smarts/nist20_merged_full_metadata_context_gnn_heads_smarts_warmup_e0_15.yml)
is also prepared, with proposal-only epochs 0–15. The SMARTS/proposal-only-warmup
construction guard is now removed and validated (2026-09-11): see the
[guard-removal finding](findings/masster/masster_code_audit.md#smarts-proposal-only-warmup-guard-removed-2026-09-11)
for the new PL-level gradient-isolation regression tests. Construction now succeeds.
Next: a real-data GPU smoke check before launching this arm.

Context-conditioning screen (2026-09-08): the user selected controlled experiments before a new
layer. The [conditioning plan](plans/conditioning_refactor_plan.md#current-controlled-screen-2026-09-08)
now selects the full-metadata, dual-CE MASSter mask1024 config as the conditioning baseline:
the user reports that NCE + ACE + adduct + instrument + mode already helps in existing runs.
The prepared NCE-only variant is an optional diagnostic outside the next run queue. Two new
FraGNNet configs complete the existing full-metadata routing pair with output scalers disabled.
Follow-up: the user requested context at the merged MASSter GNN input. The optional
`masster_context_in_gnn` path is now implemented, default off with legacy-checkpoint
migration. A matched full-metadata merged heads-only versus GNN+heads pair is prepared;
existing experiment configs are unchanged. Validation: 182 passed, 5 skipped, both merged
configs constructed, intended-only resolved diffs and Ruff passed. No runs launched.
The [pre-run design audit](findings/masster/masster_context_design_audit.md) now confirms homogeneous
groups, but all selected rows are `[M+H]+`/FT/HCD. It found ACE saturation at 100 eV in 21.42% of
selected training groups and row-order-dependent ACE selection for same-NCE replicates. Actual
GINE-EFA forward/backward checks at epochs 15/16 were finite; no quality evaluation ran.
CE fixes are now implemented: the new pair uses ACE up to 300 eV and order-invariant weighted
CE aggregation, with both models consuming the weights and legacy behavior preserved. The
[audit follow-up](findings/masster/masster_context_design_audit.md) records regression and source-data
checks. GNN initialization remains unchanged. Next: a real-data GPU smoke check before training.

FraGNNet formula-support crash follow-up (2026-09-07): W&B runs
[kvrlaxof](https://wandb.ai/frag-gnn/frag-gnn/runs/kvrlaxof) and
[71qtthvw](https://wandb.ai/frag-gnn/frag-gnn/runs/71qtthvw) both failed with infinite
loss on batch 0 before backward. Local reproduction shows that positive support targets for
unreferenced formula slots can overflow the binary loss: their logits retain the float32
minimum sentinel. The working-tree fix in `fragnnet_model.py` derives support validity from
valid joint-state contributors and zeros invalid support logits. Tensor shapes, parameters,
configs, and spectral intensity computation are unchanged. Regression coverage verifies both
attribution methods, finite gradients, max pooling, and empty support. Validation: 40 focused
tests passed on CPU; Ruff and whitespace checks passed. No full training or validation-set
evaluation ran. Next check: run a short cluster training smoke test with the patched package,
checking individual loss terms if failure persists. The exact failing remote batch was not
replayed, so this is a reproduced defect and candidate crash fix, not a confirmed remote cure.

For the next requested implementation or experiment, verify the relevant plan against code and
available artifacts. Link that task here when it spans sessions, and record its next check and
blockers in the plan. Do not duplicate the roadmap or copy its full backlog here.

## Local context caveat

The ignored `.claude/` entry point and six rule files now link to the shared guidance.
These local pointer changes are not included in Git diffs. It still contains a legacy architecture inventory;
its state directory had no files during this audit. Shared records above are the portable
entry points. Local permissions, commands, and skills were not migrated or audited for runtime
behavior. Their presence is not evidence that another checkout has them.

## Cluster submission knowledge

The [Fir/Vulcan submission guide](guides/submit_cluster_runs.md) now provides the canonical
procedure linked from INSTALL.md. Commands and profiles were checked against repository source;
no cluster connection, submission, or training validation was performed.

Connected-fragment beam review (2026-09-17): the
[reuse and equivalence finding](findings/masster/connected_fragment_beam_reuse_and_equivalence_2026_09_17.md)
records that the beam objective is `log P(mask = ∂f)` minus the complement's intact term, so it
matches `cut_masks` up to replacing the charge head with a fixed prior, ring-reconnection terms
that vanish for induced-acyclic fragments, and exact-versus-approximate search. Nine of about
thirteen search mechanisms already have MASSter counterparts; the dominant measured cost,
dense-reachability complement connectivity, is the one with an existing union-find replacement
on CPU and CUDA, which would also make the half-depth-plus-mirror growth free rather than paid
for. The finding also flags likely per-iteration host synchronization in the reachability loop.
The matching `cut_masks` inefficiency is mask-space budget: the unmerged arm enumerates 4096
masks for a 256 unique-fragment target, and that redundancy is already logged as
`unique_components_per_mask` and `duplicate_component_pathway_fraction`, so the efficiency
comparison can be answered from existing run artifacts without a new benchmark.
Source and algebra review only: no measurement, training, test run or code change. Next: the two
recorded gap checks (ring-bond probability histogram and implicit-prior versus charge-head split)
before any further beam optimization or scientific comparison.

Beam direction set (2026-09-17): keep the connected-fragment beam as a proposal-only
enumerator and restore a learned charge component, reconciling the prototype's two scores
toward its search form (the isolating mask's log probability) rather than its output form.
This removes the size/topology prior, keeps the chemically correct ring penalty through
boundary-cut count, makes complements score-symmetric again and so removes most of the
447 ms complement cost. The one structural item is emitting each mirror pair as two
components of one mask so the existing charge split and unique-component marginalization
apply unchanged. Recorded as PROPOSED in
[the FastGNN plan](plans/masster/masster_fast_gnn_proposal.md#proposed-proposal-only-beam-with-learned-charge-2026-09-17);
not implemented. The gate has now been run: at matched fragment target on identical bond logits
the beam retains the smaller side of a cut about 5:1 (fraction above half 0.167 versus 0.590),
proposes about four times fewer large fragments (0.079 versus 0.319 at or above 0.75 of the
molecule) and 1.5 times more acyclic fragments. Raising `max_boundary_cuts` from 5 to 20 leaves
the distribution unchanged and raising the fragment target makes the bias worse, so neither the
cut cap nor the budget explains it. Bond logits were synthetic; the checkpoint-logit
confirmation was not obtained. Measurements and limits are in the same finding. A budget-exhaustion run also separated ranking from coverage: fragments at or above 0.75 of the
molecule saturate near a fixed count as the budget grows 9.4x, because anything above half the
molecule arrives only as a connected mirror, so a connected fragment with a disconnected
complement is unreachable. That is a coverage hole, not a ranking effect, and the same scoring
change plus union-find closes it. The coverage-per-budget gate then closed the question against the beam. Scoring every
reference fragment by the objective the redesign would adopt, the beam's captured probability
mass saturates near 0.70 across an eightfold budget increase while `cut_masks` reaches 0.983,
and the beam costs roughly twenty times more per fragment on CPU. Decomposition by fragment size
locates the ceiling: the beam is complete below half the molecule (1.000 and 0.999 by count) and
reaches only 0.109 of fragments above 0.75 of the molecule, the band holding 47.2% of all mass.
Large fragments from a single bond cut are reachable because their complement is connected;
multi-cut large fragments generally leave a disconnected remainder and are missing by
construction. This is architectural, not a scoring artifact, so the proposal-only redesign would
not lift it. Repeating the coverage scan with **trained** bond logits from checkpoint `zlv0bli8` (epoch 41)
reproduces every direction and revises the magnitude: captured mass is flat at 0.8695 across a
fourfold budget increase against 1.0000 for `cut_masks`, so the cost of the limit is about 13% of
probability mass rather than 30%. A confident trained model concentrates mass on single-cut
masks, whose large fragment has a connected complement and is reachable; the multi-cut large
fragments remain missing. The pre-fix and post-fix beams emit identical fragment sets in 18 of 24
configurations, differing by one or two fragments in 4,001 elsewhere, so the 447 ms complement
pass buys determinism rather than materially different candidates. All measurements are
reproducible through
[scripts/debug/analyze_proposal_fragment_coverage.py](../scripts/debug/analyze_proposal_fragment_coverage.py).
Next: a decision on whether to stop the connected-fragment line; the four prepared arms should
not be trained as configured in any case.
