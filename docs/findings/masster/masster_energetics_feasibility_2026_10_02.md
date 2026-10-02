# Computed-energetics feasibility: calculations work, product-state coverage is limited

> **Status:** FINDING · **Updated:** 2026-10-02
> Authorized offline feasibility audit on frozen `i1`@59 artifacts. No model/training code,
> configs, checkpoints, processed datasets or conda environment changed. An isolated official
> xTB executable ran a small pricing sample. The ranking information gate has not run; this is
> neither an energetics null nor evidence that more measured spectra are necessary.

## Decision

Do not fit a floor-ranking correction from the present state pack. The small prediction-ranked
shortlists exclude too many winners, and a simple product-state construction covers fewer than
half of the covered floor base peaks. This would turn a ranking test into a test of a restricted
chemistry subset. The next check is target-independent product-state construction for multiple
cuts, including bond-order changes and alternative ionic states, followed by a complete frozen
shortlist extraction. Calculation cost is manageable on the audited subset.

This completes the feasibility stage of the
[energetics proposal](../../plans/masster/masster_memorization_route_plan.md#computed-product-ion-energetics-information-gate-2026-10-01-proposed).
No held-molecule spectral correction, train/calibration fit, or locked confirmation result exists.

## Shortlist retention

Re-read the complete saved base-peak ranks for 52,020 validation spectra and the 49,205-spectrum
training sample. The instrument-tolerance floor has 330 spectra / 119 molecules. Its measured
base peak is covered in **276 spectra**, distinct from the 290 spectra classified as ranking
failures by whole-spectrum support. A spectrum can have substantial covered intensity while
its base peak is uncovered.

| Frozen predicted peak shortlist | Training sample: covered base peaks retained | Floor: covered base peaks retained |
| --- | ---: | ---: |
| top 16 | 99.82% | 42.39% (117 / 276) |
| top 32 | 99.89% | 60.14% (166 / 276) |
| top 64 | 99.93% | 80.07% (221 / 276) |
| top 128 | 99.96% | 91.67% (253 / 276) |

Membership depends only on frozen prediction rank; measured winners were not inserted into a
shortlist. The validation read is exploratory and must not be described as a pristine
confirmation or a training-only choice of K. The saved peak table contains structures for
predicted top-three and measured top-three peaks, not every top-128 peak. A full top-128 ion
pack cannot be recovered from these tables alone.

## Product-state availability

Selected one distinct control spectrum for each floor spectrum, excluding all floor molecule
identities. Instrument matches exactly; nearest matching uses NCE, heavy-atom count, frozen
candidate count and novelty. There are 330 controls / 203 molecules. Absolute standardized
mean differences on those four features are at most 0.038. These are good-spectrum controls,
not a randomized intervention.

The bounded constructor preserves induced connectivity and the parent bond orders after one
Kekulization. It enumerates H placement and a localized charge edit, retains up to eight
alternative states, and accepts at most one RDKit-inferred radical electron per connected
product component. Up to 64 initial H layouts and 96 layouts including limited redistribution
are tried. Rearrangement, new bonds, alternative retained bond orders and broad spin-state
search are absent. Sanitization establishes a formal valence-compatible structure, not its
experimental identity or kinetic accessibility.

The reaction conserves the neutral parent's atoms plus one proton: product ion charge +1,
complementary product charge 0. Disconnected complementary products are separated for energy
calculations. Alternative structures behind a peak are retained; no unique latent target is
assigned.

The first sample audits 128 structures per role/selection (768 records). Only 24 / 128 sampled
floor base-peak structures have a balanced pair (18.75%). A complete follow-up checks **all**
structures behind every covered base peak in the two validation cohorts, and credits a peak
if any compatible structure has a balanced pair:

| Population | Covered base peaks | At least one ion state | At least one balanced ion + neutral state |
| --- | ---: | ---: | ---: |
| floor | 276 / 100 molecules | 199 (72.10%) | 120 (43.48%; molecule-bootstrap CI 32.50–54.18%) |
| matched controls | 330 / 203 molecules | 282 (85.45%) | 222 (67.27%; CI 60.36–73.63%) |

With both a top-128 shortlist and this bounded construction, **118 / 276** covered floor base
peaks remain (42.75%; 118 / 330 of the full floor). None of the 589 audited floor base-peak
structures with more than two boundary cuts gets a balanced pair under this method. This is a
limitation of the constructor and its fixed-bond-order/spin assumptions; it does not imply
those ions cannot exist or that energetics cannot help. Structure-level percentages and
peak-level percentages have different denominators.

**Bond-order sensitivity.** A separate RDKit bond-order-assignment check samples 64 previously
unconstructed floor structures with more than two cuts. With fixed heavy connectivity and up
to 32 H layouts, a first formal closed-shell ion + neutral proposal exists in **6 / 64** cases.
Independent explicit-H atom/charge checks pass for all six. This confirms that retaining parent
bond orders is too restrictive in some cases, but this fallback still misses most sampled
structures. It does not cover radical product states, new bonds or rearrangements, and none of
its proposals was used to fit or score spectra. A general product-state pack remains unresolved.

## Quantum calculation cost

No quantum package is available in `FRAGNNET-GPU`; CUDA is unavailable. Downloaded the
[official xTB 6.7.1 executable](https://github.com/grimme-lab/xtb/releases/tag/v6.7.1) into
`output/energetics_feasibility_2026_10_02/software/` and verified its published SHA-256:
`62a8d18778286e815292ee53d76ce447daf460a4dea3782c0f25cbac7019b5df`.
It runs without modifying the conda environment or global PATH.

Used **spGFN2-xTB** through `--tblite --spinpol`, gas phase, explicit charge and inferred
unpaired electrons, normal optimization, at most 100 cycles. Ordinary GFN2-xTB has a
spin-independent energy expression; specifying `--uhf` alone would not test spin polarization
([official explanation](https://xtb-docs.readthedocs.io/en/latest/spgfn.html)). Water and methyl
radical smoke calculations converged.

Pricing selection uses balanced **predicted top-three** channels only: eight floor, eight
matched-control and eight train-sample structures. Targets do not choose those channels. For
each, keep up to two ion, neutral and precursor alternatives, with two ETKDG conformers per
unique molecular state. Run four concurrent single-thread calculations, each with a 60-second
timeout.

- **226 unique state/conformer jobs; 210 converged, 16 did not converge** under the bounded
  settings. No embedding failure or timeout was recorded.
- Successful jobs: median **0.77 s**, 90th percentile **5.74 s**, maximum **15.98 s**. Median
  size is 22 atoms including H, maximum 74; this is a subset cost, not a full-pack extrapolation.
- Total occupied quantum slots: **566.2 s**; job timestamps span **142.9 s**. The short cached
  report replay time is recorded separately and is not calculation runtime.
- **38 / 210 converged jobs (18.10%)** change a distance-inferred connectivity graph after
  relaxation, using 1.25 times the summed covalent radii. This is a heuristic flag for possible
  rearrangement/dissociation/H migration; it does not validate bond orders or prove an invalid ion.
- Balanced electronic reaction-energy proxies can be assembled for all 24 selected channels;
  20 remain after the distance-graph filter. Their construction uses
  `E(ion) + sum E(neutrals) - E(precursor ion)`, not raw energies across different compositions.
  These values are not activation barriers, enthalpic proton affinities or a spectral ranking result.

## Precision and record correction

An independent re-read of the saved score table finds `fa1@63 - i1@63` on the `i3@63`-selected
floor is **+0.01436**, with a molecule-bootstrap interval **[-0.02228, +0.05336]** (287 spectra /
99 molecules). The older prose had reversed the subtraction; the
[canonical finding](masster_i1_worst_predictions_2026_09_30.md#result-fa1-on-the-floor-follow-up-2026-09-30)
is corrected. The interval still spans zero; the no-demonstrated-floor-gain conclusion stands.

Bootstrap SE is 0.01934. Under a simple normal planning approximation, an 80%-power effect
would be about 0.054 on that population. This is a reference from another intervention, not
the variance or power of an energetics correction. Predeclare the actual population, useful
effect, missing-state policy and numeric decision gate before fitting one; a small floor alone
may only detect large effects.

## Validation, provenance and limits

Checkout `0bbb2d1e`, with pre-existing documentation edits. Cached molecular table MD5 matches
the original extraction (`be4b5794040e68f661e6ff165dacb3b0`); SHA-256 identities and the original
extraction provenance are retained. Historical instrument labels remain in cached artifacts;
no checkpoint or dataset migration was performed for this audit.

An independent protonated-dimethyl-ether check recovers CH3+ plus methanol with the expected
mass, charge and atom balance. Full-cohort ion masses agree with saved channel masses within
2e-6 Da. Independent checks using canonical SMILES and explicit H verify **4,389 balanced
state pairs**, all 226 unique job records and all 210 converged atom identities. Scratch-script
Ruff, local documentation links and `git diff --check` pass. The initial cost report had an
object-boolean mask error after calculations finished; it was fixed and re-derived from saved
jobs without rerunning the quantum calculations. No production model tests were required.

Artifacts: `output/energetics_feasibility_2026_10_02/` contains `audit_protocol.json`, input and
software identities, environment, cohort/matching tables, shortlist retention, sampled and full
state audits, all state alternatives, quantum input/output logs, cost/proxy tables, precision
reference, validation record and scratch-source snapshots. Canonical test spectra/labels were not read.

The next product-state audit must cover multiple-cut winners and their competitors rather than
only easy single cuts. It also needs a full frozen shortlist and an independent-selector
confirmation design. Passing formal chemistry, convergence and runtime checks would authorize
a meaningful information test; none establishes that the energies improve prediction.
