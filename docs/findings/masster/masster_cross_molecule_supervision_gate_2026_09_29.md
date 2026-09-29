# Cross-molecule gate: transferable spectral relationships, weak winner labels

> **Status:** FINDING · **Updated:** 2026-09-29
> Training-only CPU census completed. Supports a bounded test of soft spectral relationship
> supervision; no MASSter training or measured model gain. Hard shared-winner labels are not supported.

## Decision

Related molecules supply nontrivial spectrum/neutral-loss relationships across held-out training
identities, beyond acquisition/mass-matched random donors. The advantage persists for 338 query
molecules whose maximum Tanimoto against the **entire** donor pool is <=0.4, in both query halves,
and for neutral losses >=50 Da. This passes the feasibility gate for informative pairs.

It does not show that the existing model fails to use those relationships. The absolute overlap
is modest, and winner agreement is weak. The next controlled experiment should preserve measured
continuous similarities, with a matched sampler control, rather than force a common bond,
fragment, or strongest-loss label on related molecules. This is distinct from the
[post-cut descriptor correction null](masster_i1_postcut_information_gate_2026_09_29.md): it uses
relationships between measured spectra, rather than descriptors to correct one checkpoint.

## Frozen design and coverage

See the [plan](../../plans/masster/masster_cross_molecule_supervision_gate.md),
[script](../../../scripts/debug/debug_masster_cross_molecule_gate.py), and standalone
[analysis configuration](../../../config/analysis/masster_cross_molecule_gate.yaml).

- Existing `nist20_inchikey` training IDs: 157,678 spectra, 12,668 molecular identities.
  Only [M+H]+, OB/OB_F and single measured NCE are eligible. Keep one spectrum per identity,
  instrument, fragmentation mode, and NCE by lowest spec ID: 153,021 distinct conditions.
- Seed 397623 splits 2,000 query identities from 10,668 donor identities. Select up to three
  conditions per query without inspecting spectra: 5,973 attempted query conditions.
- Match exact instrument, fragmentation mode, and NCE; precursor m/z distance <=max(10 Da,
  5% of query m/z). Exclude same identity and formula. Select up to three donors by radius-2,
  2,048-bit Morgan Tanimoto, without query target intensities.
- For each neighbor select three random donors, excluding the three selected neighbors and
  matching the neighbor's absolute precursor-mass distance within 2 Da. Median eligible donor
  count is 318. Neighbor/control mean mass distances are 6.756/6.762 Da; mean absolute paired
  distance mismatch is 0.601 Da.
- Scored 17,398 neighbor pairs and 52,194 random-control comparisons, spanning 1,983 query
  molecules, 5,853 query conditions, and 5,824 neighbor donor molecules. Sparse strata account
  for missing conditions. Every scored query retains some fragment intensity.
- Remove peaks at or above precursor m/z minus 1.5 Da. Compare fragment masses and neutral
  losses using untransformed intensities and one-to-one greedy product matching at 0.01 Da.
  Each profile is L2-normalized for cosine. The >=50 Da loss profile is separately normalized.
  These diagnostic cosines are **not production model cosines**.
- Average the three random controls, then average scored conditions/neighbors per query molecule.
  Bootstrap query molecules (1,000 replicates) for 95% intervals, conditional on the fixed donor
  pool. The predefined query halves have disjoint query identities but share donors.

## Results

### All scored query molecules

| Measure | Chemical neighbors | Matched random | Paired difference [95%] |
| --- | ---: | ---: | ---: |
| Fragment-mass cosine | 0.1538 | 0.0576 | +0.0962 [0.0899, 0.1025] |
| Neutral-loss cosine | 0.1084 | 0.0258 | +0.0826 [0.0771, 0.0886] |
| Neutral-loss cosine, losses >=50 Da | 0.0609 | 0.0080 | +0.0529 [0.0489, 0.0577] |
| Strongest-loss agreement | 7.20% | 1.52% | +5.68 percentage points [5.17, 6.21] |

These are relationships between measured spectra, not improvements from a new model.

### Queries unfamiliar to the entire donor pool

This stricter subset has 338 query molecules, 2,936 neighbor pairs, and 1,513 neighbor donors.
Their maximum Tanimoto to **any** donor molecule is <=0.4, even if that donor has a different
mass or acquisition condition. This is different from simply selecting a low-similarity pair.

| Measure | Chemical neighbors | Matched random | Paired difference [95%] |
| --- | ---: | ---: | ---: |
| Fragment-mass cosine | 0.1008 | 0.0509 | +0.0499 [0.0404, 0.0598] |
| Neutral-loss cosine | 0.0589 | 0.0219 | +0.0370 [0.0282, 0.0459] |
| Neutral-loss cosine, losses >=50 Da | 0.0289 | 0.0073 | +0.0216 [0.0152, 0.0281] |
| Strongest-loss agreement | 3.13% | 1.29% | +1.84 percentage points [1.14, 2.68] |

The neutral-loss advantage is +0.0396 and +0.0344 in the two query halves. The >=50 Da advantage
is +0.0242 and +0.0191. All four intervals exclude zero.

### Precursor dominance and scaffold checks

Requiring query, neighbor, and **every** random control to retain at least 20% of total intensity
after precursor removal leaves 335 globally unfamiliar query molecules and 1,935 pairs:

- fragment cosine difference +0.0664 [0.0549, 0.0785];
- neutral-loss cosine difference +0.0233 [0.0160, 0.0317];
- >=50 Da loss cosine difference +0.0197 [0.0141, 0.0259];
- strongest-loss agreement 1.10% versus 0.58%, difference +0.52 percentage points
  [−0.14, +1.21]: **no clear winner-label transfer in this strictest subset**.

Across different-scaffold neighbor pairs, the neutral-loss cosine advantage is +0.0754
[0.0688, 0.0820] over 1,817 query molecules. Empty acyclic scaffold strings count as equal.
This pair-level scaffold slice is weaker than holding out complete scaffold families.

## Limits and experiment implications

1. The global-novelty and substantial-fragmentation slices were added after the first pass;
   they are robustness reads of the same query set, not independent replication. The query
   halves and basic pair-selection recipe were fixed before the first pass.
2. Spectrum/neutral-loss agreement does not identify a bond-breaking mechanism. Different
   structures can explain the same peak. Do not turn these labels into hard fragment identities.
3. The queries are held out from the **donor pool**, but are training molecules of existing `i1`.
   No `i1` predictions were used. This is not a residual-information gate and not evidence of
   improvement beyond what `i1` already knows. The added objective must demonstrate that gain.
4. Same NCE and instrument do not perfectly fix internal energy. Stored instrument classes pool
   some instruments. The census uses single measured schedules and one replicate per condition;
   energy-ladder/replicate robustness remains a future check.
5. No correction for multiple diagnostic slices. The shared spectrum table is loaded, then filtered to training IDs before peak processing;
   validation/test labels never enter selection or scoring. No new external data, candidate
   annotations, GPU computation, or model updates.
6. The random baseline excludes the three selected neighbors. It answers whether deliberate
   chemical-neighbor sampling is more informative than the remaining eligible random pairs.
   It does not establish that any particular auxiliary loss is effective.

A bounded next experiment has three arms on one chosen current architecture: informative pair
sampler alone; the same sampler plus continuous spectral-similarity supervision; and the same
sampler with shuffled relationship targets. Keep spectrum exposure, CE distribution, optimizer
updates, and compute accounting matched. The teacher is the measured training relationship,
not a binary "same fragment" label. Promote only on held-out molecule spectrum/ranking metrics,
especially unfamiliar molecules, and confirm a promising effect with a second seed.

## Reproduction and provenance

```bash
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python \
  scripts/debug/debug_masster_cross_molecule_gate.py \
  --config config/analysis/masster_cross_molecule_gate.yaml \
  --output output/cross_molecule_gate_2026_09_29/checked
```

The checked run completed in 103.3 seconds on CPU. Git revision
`07d809de49fc6c6a43dfb3c8b0c7502009536ea4`, with existing unrelated working-tree edits preserved.
`output/cross_molecule_gate_2026_09_29/checked/` contains `pairs.pkl`, `coverage.csv`,
`summary.csv`, `provenance.json`, `audit.json`, source/config snapshots, and `run.log`.
Provenance records exact input SHA256s, source/config and tracked-diff hashes, seed, command,
Python/RDKit/NumPy/Pandas versions, time, and counts. The parent directory contains the first
pass before the added robustness columns; use `checked/` as the final result.

Validation: five focused tests passed; Ruff passed. Artifact audit confirmed all scored query,
neighbor, and control spectra are training IDs, molecular identities are disjoint between pools,
and formula matches are excluded. The standalone analysis YAML is read by this probe, not the
training runner's model-config schema.

## Pilot code disposition (2026-09-29)

The user subsequently requested reverting the cm0–2 training code. That implementation and its
launch/config/test integration have been removed; the census diagnostic, evidence and historical
artifacts remain. No full pilot run or validation gain was measured. See the
[stopped plan](../../plans/masster/masster_cross_molecule_supervision_gate.md).
