# Cross-molecule supervision: relationship census and candidate correspondence

> **Status:** FINDING · **Updated:** 2026-10-18
> Two training-only gates for supervising across related molecules: the spectral-relationship census, then deterministic candidate correspondence coverage. Never trained.

Merged 2026-10-18 from 2 findings on one topic. Each part below is the original
document unchanged except for heading levels; its own status line keeps its original date.
Former file names are listed so old links and `git log -- <former path>` can be traced.

| Part | Updated | Verdict | Former file |
| --- | --- | --- | --- |
| [Cross-molecule gate: transferable spectral relationships, weak winner labels](#cross-molecule-gate-transferable-spectral-relationships-weak-winner-labels) | 2026-09-29 | Soft relationship signal +0.050 at Tanimoto <= 0.4; hard shared-winner labels unsupported | `masster_cross_molecule_supervision_gate_2026_09_29.md` |
| [Deterministic candidate correspondence: coverage before a competition loss](#deterministic-candidate-correspondence-coverage-before-a-competition-loss) | 2026-09-29 | Coverage advantage does not predict accuracy gain; transfer plan STOPPED | `masster_candidate_correspondence_2026_09_29.md` |

## Cross-molecule gate: transferable spectral relationships, weak winner labels

> **Status:** FINDING · **Updated:** 2026-09-29
> Training-only CPU census completed. Supports a bounded test of soft spectral relationship
> supervision; no MASSter training or measured model gain. Hard shared-winner labels are not supported.

### Decision

Related molecules supply nontrivial spectrum/neutral-loss relationships across held-out training
identities, beyond acquisition/mass-matched random donors. The advantage persists for 338 query
molecules whose maximum Tanimoto against the **entire** donor pool is <=0.4, in both query halves,
and for neutral losses >=50 Da. This passes the feasibility gate for informative pairs.

It does not show that the existing model fails to use those relationships. The absolute overlap
is modest, and winner agreement is weak. The next controlled experiment should preserve measured
continuous similarities, with a matched sampler control, rather than force a common bond,
fragment, or strongest-loss label on related molecules. This is distinct from the
[post-cut descriptor correction null](masster_i1_base_peak_errors.md#post-cut-information-gate-on-i1-explicit-fragment-and-charge-descriptors-add-nothing): it uses
relationships between measured spectra, rather than descriptors to correct one checkpoint.

### Frozen design and coverage

See the plan (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_cross_molecule_supervision_gate.md`),
[script](../../../scripts/debug/masster_memorization/debug_masster_cross_molecule_gate.py), and standalone
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

### Results

#### All scored query molecules

| Measure | Chemical neighbors | Matched random | Paired difference [95%] |
| --- | ---: | ---: | ---: |
| Fragment-mass cosine | 0.1538 | 0.0576 | +0.0962 [0.0899, 0.1025] |
| Neutral-loss cosine | 0.1084 | 0.0258 | +0.0826 [0.0771, 0.0886] |
| Neutral-loss cosine, losses >=50 Da | 0.0609 | 0.0080 | +0.0529 [0.0489, 0.0577] |
| Strongest-loss agreement | 7.20% | 1.52% | +5.68 percentage points [5.17, 6.21] |

These are relationships between measured spectra, not improvements from a new model.

#### Queries unfamiliar to the entire donor pool

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

#### Precursor dominance and scaffold checks

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

### Limits and experiment implications

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

### Reproduction and provenance

```bash
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python \
  scripts/debug/masster_memorization/debug_masster_cross_molecule_gate.py \
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

### Pilot code disposition (2026-09-29)

The user subsequently requested reverting the cm0–2 training code. That implementation and its
launch/config/test integration have been removed; the census diagnostic, evidence and historical
artifacts remain. No full pilot run or validation gain was measured. See the
stopped plan (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_cross_molecule_supervision_gate.md`).

## Deterministic candidate correspondence: coverage before a competition loss

> **Status:** FINDING · **Updated:** 2026-09-29
> Endpoint and fixed-reference NCE-band training-only coverage audits complete.
> No fitting, training change, or improvement over i1 measured.

### Question and method

Can fixed ion/loss formula rules locate two observable outcomes in each of two molecules,
providing a basis for supervision of how their intensity ratio changes? This tests the coverage
needed by the candidate-transfer plan (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_candidate_transfer_plan.md`).
It does not identify shared mechanisms or independently validate fragment structures.

Use frozen `i1_inst_ob_obf` (`i50ig7rf`), epoch 59. Sample 120 query identities from the
[training-only census](#cross-molecule-gate-transferable-spectral-relationships-weak-winner-labels): 60 with maximum
donor Tanimoto <=0.4 and 60 above it. Each has one neighbor and one mass/acquisition-matched
random control, selected without intensity labels. All 360 spectra are training spectra.
Unfamiliarity here concerns the census donor pool; these are not held-out molecules for i1.

The initial condition is each query's lowest spectrum ID in the census. Generate and annotate
MASSter candidates without measured query peaks. Recover complete product-ion compositions
from atom membership, H transfer, and CRF/CMF deltas. Compare calculated ion masses to model
output; largest discrepancy was 9.15e-11 Da. Match candidate groups by exact ion formula or
nonnegative neutral-loss formula, in singly charged [M+H]+ spectra. These correspondences are
available without a learned matcher.

Sum output channels sharing a mass into one observable group. Exclude groups with conflicting
formulas, overlapping instrument-tolerance windows, and the precursor region (>= precursor
m/z minus 1.5 Da). A measured peak contributes only when exactly one candidate lies within the
instrument tolerance. Intensities below are fractions of the entire processed target; i1's
intensity transform is `none`. Two reliable outcomes means two distinct matches, each with
>=1% intensity in BOTH spectra. This is a declared coverage threshold, not a validated noise
model. The union of ion/loss mappings removes conflicting partners and can therefore contain
fewer matches than an individual view.

### Initial screen

Pairs with at least two reliable outcomes, among 60 pairs in each column:

| Correspondence | Unfamiliar neighbor | Unfamiliar random | Other neighbor | Other random |
| --- | ---: | ---: | ---: | ---: |
| Product-ion formula | 1 | 1 | 2 | 0 |
| Neutral-loss formula | 3 | 1 | 6 | 0 |
| Conflict-free union | 1 | 2 | 3 | 0 |

There are many candidate-level formula matches (about 177 loss matches per unfamiliar neighbor
pair), but very few are substantial measured peaks in both spectra. Without the 1% cutoff,
11/60 unfamiliar neighbor pairs have two positive loss matches; at a 0.1% cutoff still 11/60,
and at 0.5% it is 7/60. These are descriptive follow-ups, not alternative pass thresholds.

The initial sample is precursor-dominated: median raw intensity remaining after precursor
removal is 13.7% for unfamiliar queries and 12.2% for the other queries. Consequently this
screen does not establish that competition pairs are scarce at more fragmenting energies.
A follow-up on the SAME query identities selects the highest NCE available in the matched
census, without inspecting intensities. It is a sensitivity check on the same molecules,
not an independent replication; it does not necessarily select the highest NCE in all NIST.

Of 239,435 initial candidate groups, 227,758 survive eligibility rules; 10,957 have overlapping
mass windows, none have conflicting elemental formulas, and 143,777 contain multiple retained
atom sets. Structural ambiguity therefore cannot be ignored even though outcome formulas are
consistent. Formula matches must not be promoted to claims of shared structure or mechanism.

### Artifacts and implementation

- [Audit script](../../../scripts/debug/masster_memorization/debug_masster_candidate_correspondence.py),
  [initial config](../../../config/analysis/masster_candidate_correspondence.yaml), and
  [higher-NCE config](../../../config/analysis/masster_candidate_correspondence_high_nce.yaml).
- `output/candidate_correspondence_2026_09_29/`: selected pairs, cached candidate groups,
  individual correspondences (including zeros), pair coverage, summaries, source/config
  snapshot and provenance. Initial CPU run completed in 170.4 seconds.
- `threshold_sensitivity.csv` and `query_conditions.csv` are post-run descriptive reads of
  saved matches/census metadata. They did not change selection or matching.
- `high_nce/` holds the subsequent condition-sensitivity run, including its own source snapshot.
  Source changes after the initial run add the condition-selection option and compute union
  degrees with counters; the original source and original config are preserved with hash checks.
- Tests cover water-loss correspondence, ambiguous masses/formulas, alternate structures,
  precursor/negative-loss exclusions, union conflicts, target-independent matching, query
  selection, coverage thresholds, and real-model formula/mass reconstruction.

### Higher-NCE condition sensitivity

On the same 120 query identities, selecting the highest NCE in the matched census yields:

| Correspondence | Unfamiliar neighbor | Unfamiliar random | Other neighbor | Other random |
| --- | ---: | ---: | ---: | ---: |
| Product-ion formula | 45/60 | 29/60 | 46/60 | 29/60 |
| Neutral-loss formula | 7/60 | 1/60 | 19/60 | 2/60 |
| Conflict-free union | 31/60 | 28/60 | 32/60 | 26/60 |

The same >=1% threshold applies at both endpoints of each of two matches. Mean unfamiliar
query intensity covered by ion matches is 58.3% for neighbors and 40.4% for random controls;
for loss matches, 23.1% versus 9.4%. These are coverage measurements, not predicted cosine
or evidence that the measured intensity ratios will improve a model. The union's loss of
coverage reinforces keeping the ion and neutral-loss views separate in the next test.

This is a very high-energy sample: median NCE is 157.5 for unfamiliar queries and 160 for
the other queries; median nonprecursor intensity fraction is 1.0 in both. The selected
spectrum changes for 117/120 queries; instrument labels do not change. This favors small,
common product ions and cannot establish coverage at moderate NCE.

The high-NCE choice is a post-initial-screen condition sensitivity on the same molecules,
selected using acquisition labels only. References can change with the selected condition.
It is not a replication and does not establish an optimal training CE policy.

### Decision

The deterministic matcher is operational, and higher-NCE ion-formula matches provide a usable
pool of two-outcome cases without learned graph matching. However, substantial coverage also
exists for random molecules. Shared small product formulas can be generic rather than
mechanism-specific. Formula correspondence establishes comparable outcomes, not shared
structural pathways or transferable allocation rules.

Proceed to a bounded incremental-information test using separate ion and loss views, with
matched random-reference, reference-structure-only, and shuffled-intensity controls. The
competition loss itself remains unimplemented: first establish that reference intensities help
correct i1's held-out errors. The initial weak-coverage result remains valid for its sampled
conditions; the sensitivity shows why it cannot be generalized across energies.

Both artifact audits verified source/config or selection hashes and formula-equality/eligibility
of saved correspondences. The second CPU run processed 358 unique spectra in 160.5 seconds.

Nine focused tests pass; the five census tests also passed before the condition-selection
extension. Ruff passed. No training was launched and no change was made to staged cm0–2 code.


### Fixed-reference NCE-band audit (complete)

A further audit addresses the endpoint-selection limitation. Use the same original 120 queries,
fix each original neighbor/control identity, and select a common observed NCE near the center
of each band. Instrument/adduct/fragmentation mode stay fixed, with no interpolation or
replacement donor to fill gaps. Selection uses the full training condition table, not the
census's three conditions per query. The bands were frozen before reading matching outcomes.

| NCE band | Unfamiliar available / 60 | Other available / 60 |
| --- | ---: | ---: |
| [0,20) | 36 | 32 |
| [20,40) | 51 | 50 |
| [40,60) | 43 | 50 |
| [60,100) | 45 | 41 |
| [100,201) | 34 | 39 |

Only 15 queries in each novelty stratum have data in all five bands. Report both available-case
coverage and the fixed common cohort, with instrument-specific results retained. This design
isolates reference identity across bands but does not make NCE physically interchangeable
between instruments. Sparse common-cohort estimates are descriptive.

[Selector/summarizer](../../../scripts/debug/masster_memorization/debug_masster_correspondence_nce.py),
[config](../../../config/analysis/masster_candidate_correspondence_nce.yaml), artifacts
`output/candidate_correspondence_2026_09_29/nce_bands/`. 842 spectrum pairs / 1,250 unique spectra.
Twelve focused tests and Ruff passed. Annotated CPU inference completed in 392.8 seconds;
maximum formula/mass discrepancy remained 9.15e-11 Da. The first two audits remain separate,
because their reference identities can differ from this fixed-triple design.


#### Available-case results for unfamiliar training queries

Counts of query/reference pairs with at least two distinct formula matches, each >=1% intensity
in BOTH spectra. Neighbor and random counts use identical query cases within each band.

| NCE band | Queries available | Ion formula: neighbor vs random | Loss formula: neighbor vs random |
| --- | ---: | ---: | ---: |
| [0,20) | 36 | 0 vs 0 | 3 vs 0 |
| [20,40) | 51 | 3 vs 0 | 6 vs 1 |
| [40,60) | 43 | 12 vs 3 | 7 vs 0 |
| [60,100) | 45 | 15 vs 6 | 6 vs 1 |
| [100,201) | 34 | 23 vs 20 | 2 vs 1 |

Median selected NCEs are respectively 10, 30, 50, 75 and 160. Thus strong ion-formula
competition pairs exist at moderate NCE: 27.9% versus 7.0% at [40,60), and 33.3% versus 13.3%
at [60,100). The highest band yields more matches but also much more random-reference
coverage (67.6% versus 58.8%). These are descriptive differences, not inferred improvements
in prediction or formal evidence of learned chemistry. Instruments and reference identities
are fixed within each query trajectory; available-case query composition differs by band.

#### Same 15 unfamiliar queries in every band

| NCE band | Ion formula: neighbor vs random, out of 15 | Loss formula: neighbor vs random, out of 15 |
| --- | ---: | ---: |
| [0,20) | 0 vs 0 | 0 vs 0 |
| [20,40) | 0 vs 0 | 0 vs 1 |
| [40,60) | 3 vs 1 | 1 vs 0 |
| [60,100) | 6 vs 1 | 1 vs 0 |
| [100,201) | 11 vs 8 | 0 vs 0 |

[Coverage plot](../../../output/candidate_correspondence_2026_09_29/nce_bands/nce_coverage.png)
(PNG; SVG also saved). This small complete-case cohort supports the qualitative ion-coverage
pattern without changing molecular triples. Loss-pair coverage is sparse here; the larger
available-case counts must not be read as a within-molecule energy trend. Complete-case
availability itself selects particular molecular/measurement schedules.

`nce_summary.csv` includes both novelty groups, all three matching views, and OB/OB_F-specific
summaries. `availability.csv` preserves missing bands. `audit.json` checks fixed donor identities,
exact acquisition matches, formula-key equality, unique matched endpoints per view, and
source/config/selection identities. The saved selection source predates a plot-title-only
change; `summary_source.py` records the final summarizer used for the chart.

#### Updated next decision

Extreme NCE is unnecessary for finding some usable outcomes. Prioritize a bounded information
check on ion-formula matches in [40,100), retaining the random/structure-only/shuffled controls
and reporting supported coverage. Keep neutral-loss matching as a separate, sparser view.
The band choice is informed by this coverage audit and requires independent evaluation later;
it is not an optimized or validated training policy. Do not add the competition objective yet:
we still need evidence that reference intensities explain i1 errors beyond its own prediction.
No model fitting, new training run, or changes to staged cm0–2 were made.

### Incremental-information follow-up (2026-09-29)

The fresh validation-query gate (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_reference_information_gate_2026_09_29.md`) is complete
and negative for the tested correction. Ion alignment selects zero residual in both folds; loss
alignment degrades cosine. The coverage findings above remain valid, but they do not justify
promoting this candidate-transfer recipe. See the linked result for controls and limitations.
