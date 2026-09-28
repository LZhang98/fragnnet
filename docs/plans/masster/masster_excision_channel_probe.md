# MASSter excision continuation probe

> **Status:** COMPLETED · **Updated:** 2026-09-28 · Diagnostic only; no model channel built.

## Objective and decision

Test whether deleting a small internal bridge X from c0's existing connected candidate S
adds observed peak coverage beyond its pool and current whole-precursor templates.
Generate connected, acyclic X with at most three heavy atoms, exactly two external bonds,
and distinct external endpoints a and b. Only use S containing X, a, and b.
This represents connected products spanning the newly formed a-b bond; products retaining
only one endpoint are ordinary connected subsets. The equivalence concerns atom sets,
not reaction feasibility, hydrogen state, or candidate ranking.

## Checks and protocol

1. Test the identity against explicit rewiring on small graphs, including ring rejection,
   one-ended fragments, and a post-excision cleavage that whole-precursor templates miss.
2. Extend `debug_masster_smarts_rule_gate.py` with a continuation probe entry point. Reuse
   model channels, random disconnected controls, and molecule-bootstrap summaries.
3. Preserve c0's original metadata and strict weight loading. The current data distinguish
   OB/OB_F but c0 has one historical FT embedding: alias its config label to OB and feed
   both Orbitrap classes index zero. No dataset files are relabelled by this probe.
4. Sample molecules on train, with a seeded bounded sample of spectra per molecule.
   Group bridges by local environment. Confirm only frozen train-selected groups on val;
   never inspect test. Report multiple-testing and small-class limitations.
5. Baselines: retained pool; pool plus current templates; additionally all generic
   whole-precursor bridge products. Separate generic whole-product reach from continuation.
   Filter matched continuation channels against connected same-composition/H alternatives.
   Capped searches are unresolved, never evidence of unique coverage.
6. Existing valence channels count broken precursor bonds. Report both that optimistic
   convention and a single-new-bond correction consuming two boundary valence slots.
   Neither establishes a physical reaction mechanism.

Before validation, freeze an additional multiplicity check: use 20,000 paired molecule
bootstrap draws (seed 20260928), with two-sided percentile limits at 0.05/(2K) and
1-0.05/(2K), where K is the number of train-selected families. Require at least ten
validation molecules, positive adjusted lower limit, and no unresolved real/null queries
or missing null samples for a family-level confirmation. Report ordinary intervals too.

Acceptance is additional coverage beyond the current templates and a size/count-matched
random disconnected null, confirmed on held-out molecules. A coverage pass motivates a
channel design; it is not evidence of a cosine gain. A null ends this proposal at the probe.
The proposed 8% prevalence, 0-2 channels, and <0.005 cosine gain are unverified hypotheses
for generic bridge enumeration, not assumptions of this run.

## Provenance and execution

Checkpoint: `saved_ckpts/c0_control_vcpho5qm.ckpt` (epoch 42, historical `[M+H]+`/FT only).
Current code and processed NIST20, saved `nist20_inchikey` train/val splits, FRAGNNET-GPU.
The runs saved resolved config, arguments, Git revision/diff identity, input identities,
sample IDs, timing, and per-spectrum/per-family measurements under `output/`.
No negative-mode conclusion can be drawn from this checkpoint.

## State

Diagnostic completed. Seventeen scientific tests pass; c0 loaded strictly. Train screened
512 molecules / 2,039 spectra, and frozen-family validation scored 512 / 2,035. The nitro
NO-loss family confirms after multiplicity correction; the generic train union is not
clearly above random. NH and carbon validation have unresolved queries. See the canonical
[finding](../../findings/masster/masster_excision_continuation_gate_2026_09_28.md) for numbers,
limitations, literature follow-up, and the restricted channel recommendation. No production
channel was built; this closes the diagnostic, with channel implementation and training
remaining separate work.

An initial run was stopped because exhaustive connected-H enumeration hit its cap even
when a sufficient twin could already have been found. Its partial artifacts are preserved as
`output/excision_continuations_2026_09_28/initial_capped_train.*`. The completed rerun uses a
tested query-specific early exit; capped queries remain unresolved and cannot count as unique.

Command (FRAGNNET-GPU interpreter):

```bash
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python scripts/debug/debug_masster_smarts_rule_gate.py continuations \
  --output output/excision_continuations_2026_09_28/train.pkl \
  --molecules 512 --spectra-per-molecule 4 --batch-size 4
```

Run artifacts include sample IDs, resolved configuration, source hashes, saved source copies,
and per-spectrum/per-environment rows. The current device is RTX 2070 SUPER (8 GB), with
CPU chemistry checks. This c0 probe is positive-mode only. Seventeen unit tests, Ruff, and the
checkpoint smoke passed; the restricted nitro gate confirmed.
Related: [literature](../../notes/masster_excision_literature_2026_09_28.md),
[current template gate](../../reference/masster_smarts_rules.md).

Validation command, after freezing `train.families.json`:

```bash
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python scripts/debug/debug_masster_smarts_rule_gate.py continuations \
  --split val --families output/excision_continuations_2026_09_28/train.families.json \
  --output output/excision_continuations_2026_09_28/val.pkl \
  --molecules 512 --spectra-per-molecule 4 --batch-size 4 --seed 1
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python output/excision_continuations_2026_09_28/confirm_families.py
```
