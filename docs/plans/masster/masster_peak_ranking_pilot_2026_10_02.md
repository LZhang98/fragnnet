# MASSter explicit peak-ranking continuation pilot

> **Status:** CLOSED NULL · **Updated:** 2026-10-02 · CPU pilot complete: tuning selects zero
> ranking weight. No end-to-end training conclusion or cluster submission.

## Objective and scope

Test whether an explicit measured-base-versus-strongest-incorrect-peak preference loss adds
held-out signal beyond the ordinary spectrum loss. The existing
[base-peak analysis](../../findings/masster/masster_i1_base_peak_errors_2026_09_28.md) identifies
covered ranking failures; the
[cut-probe null](../../findings/masster/masster_cut_intensity_probe_2026_10_01.md) tested a different,
additive frozen correction and does not answer this training question.

CUDA is unavailable on this workstation. This pilot freezes the encoder and proposal model and
fine-tunes only the existing intensity head in eval mode (dropout and batch statistics frozen).
It tests a continuation signal, not end-to-end training or new-molecule generalization from
scratch. Candidate support stays fixed. No canonical test targets are read.

## Locked protocol

- Start every arm at `i1` epoch 59, strict weight loading. Same ordered batches, optimizer,
  learning rate, train identities, horizon and initial parameters.
- Sample 256 training identities from the archived 4,000-molecule train sample and 512 validation
  identities from all validation, at most three conditions each, before looking at intensities.
  Audit full `inchikey_s` separation; divide validation identities equally into tuning and
  confirmation. These validation molecules were inspected historically and are held out only
  from this pilot's strength selection.
- Train four epochs with ordinary production loss plus ranking weights 0, 0.1 or 0.3.
  Sum channels at four-decimal mass, then sum all base-compatible masses for the positive group.
  The negative is the strongest incompatible mass group. Auxiliary is measured top-two relative
  gap times `softplus(log P_negative - log P_positive)`; skip uncovered bases and gaps <0.1.
  Reduce over all spectra, including zero-loss skips. Evaluation excludes no ambiguous/tied cases.
- Select weight on tuning instrument-tolerance cosine; ties prefer smaller weight. Read the
  locked confirmation contrast against both unchanged checkpoint and matched weight-zero
  continuation. Report base hits, coverage, compatible base rank, NCE/instrument slices and
  molecule-clustered confidence intervals. Fixed final horizon; no epoch selection.
- A useful pilot requires positive confirmation cosine and base-hit differences against the
  matched control, with cosine's 95% interval excluding zero. A practical +0.005 cosine gain is
  the existing single-seed campaign screen, not a powered test threshold for this small sample.
  A pilot null cannot rule out full-data/end-to-end training. Nested-pair and novelty-specific
  claims require a subsequent annotated evaluation; this pilot does not identify fragment paths.

## Execution and provenance

Initial working tree has unrelated documentation edits in `docs/README.md`, `docs/project-state.md`,
the JEPA plan and memorization-route plan. Preserve them. The project-state statement that the
Orbitrap rename is uncommitted is stale relative to the inspected code status.

Parameters: [analysis YAML](../../../config/analysis/masster_peak_ranking_pilot.yaml).
Entrypoint: [pilot script](../../../scripts/debug/masster_memorization/debug_masster_peak_ranking_pilot.py).
Outputs: `output/peak_ranking_pilot_2026_10_02/`, including sampled manifest, strict input/source
identities, resolved config, Git revision/diff identity, loss history, final head weights,
per-spectrum scores, locked selection, confidence intervals and energy/instrument slices.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python \
  scripts/debug/masster_memorization/debug_masster_peak_ranking_pilot.py
```

## Results and next check

All arms completed in 417.9 seconds with finite losses/gradients. Actual sampled counts are
765 training, 762 tuning and 763 confirmation spectra; each partition has 256 identities.
Tuning selects weight zero. Added weights 0.1/0.3 lower confirmation cosine by 0.000221/0.000758
against matched ordinary-loss continuation, despite improving training ranking. Coverage is
identical. The small confirmation base-hit increases do not reproduce on tuning.

Six objective/sampling tests and thirteen inherited scorer/probe tests pass; Ruff, whitespace,
identity/membership/coverage and source-hash audits pass. No full model suite or GPU test ran
for this standalone diagnostic. See the
[canonical result, intervals, slices and limits](../../findings/masster/masster_peak_ranking_pilot_2026_10_02.md).
Preserve the pilot null; any full-data/end-to-end follow-up must distinguish that untested scope.
