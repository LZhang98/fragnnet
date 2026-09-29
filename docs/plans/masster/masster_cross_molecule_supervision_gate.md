# Cross-molecule supervision feasibility gate

> **Status:** STOPPED · **Updated:** 2026-09-29
> The user requested removal of the cm0–2 code. Training integration, configs, launcher,
> manifest preparation and pilot tests are reverted. Census findings and artifacts are retained.

## Question and scope

Do related training molecules share measurable fragmentation patterns beyond matched random
molecules? This tests whether informative cross-molecule supervision pairs exist. It does not
measure improvement over MASSter, establish reaction mechanisms, or revive the almost-null
[local bond learner](../../findings/masster/masster_bond_local_learner_2026_09_24.md).
The reopening audit
is the design source. `fa1` and `mz1` are running on W&B per the user; their results are pending.

## Frozen first screen

- Use the existing NIST20 training spec IDs, [M+H]+, OB/OB_F, single measured NCE schedules.
  Hold out 2,000 training molecular identities as queries; all other training identities are
  donors. Never use validation/test spectra or query spectra to select chemical neighbors.
- Match exact instrument, fragmentation mode, and measured NCE. Restrict precursor m/z distance
  to max(10 Da, 5% of query m/z). Exclude same molecular identity and same formula.
- Keep one spectrum per molecule/acquisition condition, chosen by spec ID, and at most three
  conditions per query molecule, spread over the sorted available conditions.
- Select the three most similar eligible donors by radius-2, 2,048-bit Morgan Tanimoto. Compare
  each against three random donors matched additionally to its precursor-mass distance within
  2 Da. If fewer than three controls exist, record the exclusion rather than weaken matching.
- Remove peaks at or above precursor m/z minus 1.5 Da in both spectra. Compare the remaining
  absolute fragment-mass spectra and neutral-loss spectra with one-to-one 0.01 Da tolerance
  cosine on untransformed intensities. Also report neutral losses >=50 Da to detect dependence
  on common small losses. These are diagnostic scores, not production MASSter cosine.
- Report top-loss agreement, donor support, mass balance, and similarity/scaffold slices.
  Bootstrap differences by query molecule; multiple energies and neighbors are not independent
  samples. Freeze two disjoint query halves for a directional replication read.

## Additional robustness read

After the first pass showed positive pair signal, add descriptive checks for (1) query maximum
Tanimoto against **all** donor identities, not only acquisition/mass-eligible donors, and (2)
queries, neighbors, and every random control retaining at least 20% of total intensity after
precursor removal. The second subset checks that renormalized weak residual spectra are not
the sole source of the effect. These extra slices are follow-ups on the same query set, not an
independent replication. Empty acyclic scaffolds count as equal, not as different scaffolds.

## Interpretation and next decision

A pass requires positive neighbor-minus-control signal on both query halves, including the
low-similarity population and beyond common small losses, plus enough distinct donor support.
A close-analogue-only gain does not justify claiming improved novel-scaffold generalization.
A census pass justifies a bounded auxiliary-supervision experiment, not an expected cosine gain.
Its controls must include the same sampler without the objective and shuffled relationships.
No pass means no trainer implementation from this premise; retain the negative evidence.

## State and validation

Completed the [probe](../../../scripts/debug/debug_masster_cross_molecule_gate.py) and five
focused tests. CPU census and artifact audit passed; Ruff passed. Canonical results and limitations
are in the [finding](../../findings/masster/masster_cross_molecule_supervision_gate_2026_09_29.md).
The final artifact directory is `output/cross_molecule_gate_2026_09_29/checked/`.

The continuation uses `i1` as the established anchor while `fa1`/`mz1` run. Their pending
results do not enter the three-arm comparison below. The feasibility result does not justify
hard common-winner supervision or predict a cosine gain.


## Historical training pilot implementation (code reverted)

The implementation described below was removed at the user’s request on 2026-09-29.
This section preserves the historical design and checks; its launcher instructions are inactive.
The user originally requested continued work after the positive census. This was a bounded test of a
continuous relationship objective, not a claim that the census found information missing from
`i1`.

### Representation and objective

- Optional two-layer projection of the mean precursor atom embedding concatenated with the
  existing spectrum context. Hidden width 64, SiLU, output two 64-dimensional vectors: one for
  absolute-fragment similarity and one for neutral-loss similarity. Gradients reach the shared
  precursor and context encoders. The spectral decoder continues to train on its normal loss.
- For a manifested pair `(i,j)`, predict the cosine between each pair of projection vectors.
  Regress the two measured similarities with mean squared error. Loss weight 0.1 is a fixed
  pilot choice; no tuning on validation data has been performed.
- Average over pairs/channels within each batch. Split pair losses equally between endpoints
  and scale by `batch_size / (2 * pair_count)`, so the existing unweighted batch reducer returns
  the pair mean rather than diluting it according to the number of participating spectra.
  No-pair batches have differentiable zero loss. Log loss and endpoint degree.
- The branch is train-only; validation/inference does not need the pair manifest. Historical
  checkpoints migrate to disabled defaults, and default-off model outputs/state shapes are
  preserved. Forked initialization leaves existing parameters unchanged. All three pilot arms
  instantiate the same projection so initial weights and forward computations match.

### Frozen manifest and exposure

`prepare_masster_relationship_pairs.py`
revalidates the census source hashes, reproduces neighbor scores, and computes individual
control-pair scores. It writes 68,342 unique undirected pairs to
`data/auxiliary/masster_relationship_2026_09_29/pairs.csv` (derived artifact, Git-ignored).
SHA256: `552925a84e64840cfebcf0f8a0785fa3afc87256bd0caae59d90c1fd07e8248d`.
The adjacent JSON records sources and preparation details. The checksum is pinned in each config.

Shuffle the joint fragment/loss target vectors once within instrument, adduct, fragmentation
mode, measured NCE, and neighbor/control kind. A seeded permutation followed by a cyclic shift
avoids fixed row assignments in groups of size >1. Singleton strata would retain their own label; the generated manifest has zero singleton
shuffle groups. This preserves the stratum target distribution and uses
no validation/test spectra.

The pair sampler randomly interleaves neighbor and control edges, greedily matches disjoint
endpoints, randomly pairs remaining spectra into filler units, and shuffles all units. An even
batch size keeps selected pairs together. Every training spectrum appears exactly once per
epoch, including the last incomplete batch; no oversampling of donor hubs or molecule/CE
exposure change. All manifested pairs co-occurring in a batch contribute to the loss. Existing
epoch seeding applies to the sampler's generator. Require single-device, unmerged, unweighted
training with unique spectrum IDs; incompatible samplers and drop-last are rejected.

### Matched arms

| Arm | Relationship target | Weight | Control |
| --- | --- | ---: | --- |
| cm0 | measured, logged only | 0 | sampler/projection control |
| cm1 | measured | 0.1 | cm0 |
| cm2 | permuted within strata | 0.1 | cm1 and cm0 |

All start from the same `i1` recipe and seed 397623, 64 epochs, batch 512, corrected OB/OB_F
labels, and the existing spectral objective. The concurrently developed retained-fragment
branch is explicitly disabled in these configs. Its source edits are preserved.

The read is cm1 versus cm0 and cm2 on production validation cosine and unfamiliar-molecule
ranking, followed by a second seed for a promising result. Also compare cm0 against the existing
`i1` anchor to price the sampler change. Lower auxiliary training error is not a pass. Do not
compare the two-step smoke losses as an accuracy result.

### Execution and validation

- Nine focused tests cover checksum/ID errors, pair lookup, exact once-per-epoch exposure,
  deterministic reseeding, pair-loss normalization, both-endpoint gradients, empty pairs,
  historical checkpoint compatibility, train-only behavior, actual loss wiring, and manifest
  staging. Existing MASSter/config/job tests also exercised the integration.
- `smoke_masster_relationship.py` runs
  identical actual batches with identical initial parameter hashes in all three arms, checks
  finite optimizer updates and nonzero auxiliary gradients, and writes resolved configs,
  source/diff hashes, batch IDs, timings, losses, and memory. CPU smoke completed six updates
  (two per arm, batch 32); CUDA smoke also completed six finite updates, with nonzero auxiliary gradients in cm1/cm2
  and zero auxiliary gradients in cm0. Peak allocated CUDA memory was at most 1.25 GiB at
  batch 32; this does not establish batch-512 memory or full-run throughput. The selected smoke batches must contain
  pairs, so timings are not a random full-epoch performance benchmark.
- The cluster generator stages the new manifest. The
  Fir launcher prepares
  all three jobs with `--no_submit=True`. Regenerate on the Fir login node after transferring
  the code, configs, and derived CSV. No Slurm executable is available in this local workspace.
- No 64-epoch job submitted in this session. No checkpoint has been promoted. Outputs are under
  `output/cross_molecule_gate_2026_09_29/`.

### Final preparation audit (2026-09-29)

The batch-512 sampler audit covered all 157,678 training spectra exactly once across 308
batches. This seed produced 6,037 in-batch relationships (8–35 per batch, mean 19.60), with
no empty batches. This is a subset of the 68,342-edge manifest each epoch; randomized matching
changes the subset across epochs. The auxiliary head adds 41,152 parameters (about 0.45%),
for 9,256,853 total in each arm. Resolved configs differ only in loss weight and shuffle flag,
apart from run labels. Three dry-run Fir scripts were generated and their manifest staging
verified under `output/cross_molecule_gate_2026_09_29/jobs/`.

Artifacts: `smoke_cpu.json`, `smoke_cuda.json`, `sampler_audit.json`, and `pilot_sources.json`
under the same output directory. Next: transfer the derived manifest and code to Fir, regenerate
the three jobs using the launcher, and run the matched pilot. Validation spectral quality has
not yet been measured.

Final regression checks: 235 of 237 tests passed on the combined run; the two failures caught
relationship keys placed in the FraGNNet template section. After moving those declarations
to the MASSter section (no value changes), all 22 config tests passed. Together the checks
cover all 237 cases. Ruff, shell syntax, local plan/finding links, and `git diff --check` passed.
The full repository test suite was not run.

## Revert at user request (2026-09-29)

Removed the cm0–2 configurations, relationship pair sampler/lookup and loss modules, pair-manifest
preparation and optimizer-smoke scripts, Fir launcher, and pilot-specific tests. Removed their
shared model/PL/runner/template/checkpoint-migration/job-generator integration. Retained the
training-only census diagnostic and its tests, historical findings and generated artifacts,
and all separate candidate-alignment and retained-fragment work. No full pilot run had been
submitted. This stop is a user-requested code rollback, not a measured cm1 validation null.

Rollback validation: 486 config, runner, cluster-generator, MASSter model/PL and retained-fragment
tests passed. Ruff and `git diff --check` passed. No CM runtime/config references remain under
`src/`, `scripts/`, `tests/` or `config/`; the Git index remained empty. A local recovery copy of
the removed code and exact reverse patch is under `/tmp/frag_gnn_cm_revert_20260929/`.
