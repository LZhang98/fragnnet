# Computed-energetics feasibility: formal coverage improves; geometry and information remain unproven

> **Status:** FINDING · **Updated:** 2026-10-02
> Authorized offline feasibility audit on frozen `i1`@59 artifacts. No model/training code,
> configs, checkpoints, processed datasets or conda environment changed. An isolated official
> xTB executable ran two pricing/geometry samples. The ranking information gate has not run; this is
> neither an energetics null nor evidence that more measured spectra are necessary.

## Decision

The continuation resolves much of the **formal construction** limitation: a joint search over
H placement, bond orders, charges and radicals, combined with the original states, covers
268 / 276 covered floor base peaks (97%). Complete prediction-ranked top-128 shortlists and
full frozen channel outputs now exist. Together with top-128 selection, 245 / 330 full-floor
spectra have a formal balanced state, up from 118 / 330. This is a feasibility improvement,
not an improvement in spectral prediction.

Do not equate formal coverage with valid optimized products. Only 14 / 24 prediction-selected
multiple-cut channels retain a distance-preserved ion + neutral pair in the second geometry
audit. Before fitting a spectral correction, settle a bounded, target-independent state/geometry
policy and its calculation budget. Keep unresolved channels and spectra in evaluation with
explicit missing-feature accounting; excluding them would bias the gate. The next scientific
test is incremental information from energies and calculation-status features beyond existing
descriptors, with separate calibration, independent-selector confirmation and shuffled controls.
The first audit and its restrictive-constructor results below remain historical evidence.

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

## Continuation: joint states and complete frozen outputs

The user authorized continuation. A scratch-only integer search jointly selects total H
placement, single/double/triple orders on retained edges, localized formal charges and radicals.
Its objective minimizes structural edits; it is **not an energy function**. It preserves
heavy-atom connectivity and isotopes, limits each atom to at most one radical, enforces net
charge +1 for the connected ion and 0 for each neutral component, and checks electron parity.
Two searches allow zero or at most two radicals per component. Retain at most two sanitized
SMILES per product/search, with a two-second limit per solver call. No new bonds, rearrangement,
stereoisomer enumeration or ground-state spin-coupling assignment is supplied.

Element/charge/valence templates are bounded proposals validated by RDKit, whose
[allowed-valence rules](https://www.rdkit.org/docs/RDKit_Book.html#valence-calculation-and-allowed-valences)
are formal constraints rather than evidence of a fragmentation mechanism. Joint enumeration
avoids the earlier first-32/64 H-layout ordering limitation.

- On the **same 64** previously missed multiple-cut structures used by the bond-order check,
  the joint search supplies 36 closed-shell balanced proposals and 51 with radical allowance,
  versus six from the previous fallback. A first exploratory pilot sampled a different 64
  and found 33/44; the matched-sample replay is the reported comparison.
- The full fallback audits 927 distinct previously unbalanced molecule/atom-set/H-shift
  channels: 669 receive a closed-shell pair, 806 receive a pair with radical allowance.
  Elapsed construction time is 109 s on this subset, including alternative enumeration.

| Population | Original constructor | Original states plus closed-shell fallback | Original states plus radical-allowed fallback |
| --- | ---: | ---: | ---: |
| covered floor base peaks | 120 / 276 (43.48%) | 250 / 276 (90.58%) | 268 / 276 (97.10%) |
| matched-control base peaks | 222 / 330 (67.27%) | 319 / 330 (96.67%) | 329 / 330 (99.70%) |

The middle column retains the original states, some of which contain radicals; it is not a
claim that all 250 winners have closed-shell products. With top-128 selection, the two unions
retain 229 and 245 covered floor winners respectively. The latter is 88.77% of the 276 covered
winners, or 74.24% of all 330 floor spectra. This includes no quantum-validity guarantee.

**Complete frozen shortlist.** Fresh CPU inference on all 660 cohort spectra / 322 molecules
strictly loads the original checkpoint weights into the same architecture. A spectrum-table
copy renames historical instrument labels while preserving their embedding order; the original
checkpoint/data remain untouched. The exporter saves 87,462 peak rows, including diagnostic
target-selected peaks. The actual prediction shortlist explicitly filters `pred_rank <= 128`:
**84,368 peaks, 345,746 structure rows, 199,903 unique molecule/atom-set/H-shift/mode channels**.
No measured winner is inserted. All alternatives remain in the stored shortlist. Restricting
calculation to the one/two highest predicted-share alternatives would still involve
52,014/90,001 unique channels; this is a budget estimate, not an adopted chemistry policy.

A second frozen pass stores all **1,238,427 output channels** and target arrays separately
for scoring. Production one-to-one instrument-tolerance cosine agrees with a direct helper
on 32 spectra to floating-point precision. Against the earlier full CPU cut-probe read,
all base-peak hits and all covered best-compatible ranks agree; cosine differs by at most
0.002489, mean difference -0.00000698. Against the older CUDA export, 17 spectra have changed
candidate counts, with a maximum count difference of 16. Neither is a bit-identical replay.
Use this fresh baseline and pack together for any correction comparison.

Every fresh nearest-compatible covered base-peak structure reconciles with the constructed
state cache: no unknown channels, the same 268/329 formal coverage, and the same 245 floor
winners retained at top-128.

**Exporter metric limitation.** Its many-to-one target mapping can assign one measured peak
to two nearby candidate masses. The fresh export reaches an `oracle` of 1.41295: spectrum
452323 assigns its base intensity to both 145.063449 and 145.064791 Da. Its decomposition
cannot serve as a bounded support oracle or efficacy metric in this audit. Stored predicted
ranks/structures remain useful. No production metric or exporter code was changed; the
complete-channel baseline uses production one-to-one matching. This does not invalidate the
earlier cut-probe result, which independently verified that matching convention.

**Harder geometry sample.** Choose 12 distinct floor molecules and 12 matched-control
molecules using only prediction-selected top-128 multiple-cut channels; targets do not select
channels or states. The search tries 27 channels to obtain 24 balanced proposals, with
three to eight cuts and predicted ranks 1–118. Keep up to two ion, neutral and precursor
alternatives and two ETKDG conformers. Reuse the verified spGFN2-xTB executable and settings.

- 322 unique jobs: **297 converge, 25 do not**. Successful median 0.72 s, p90 4.62 s,
  maximum 10.51 s. Size median 20 atoms including H, maximum 59. Occupied quantum slots
  total 628.6 s; selection, construction and calculations complete in 196.6 s.
- 76 / 297 converged jobs (25.59%) change the distance-inferred graph. The heuristic is
  unchanged; it cannot validate bond orders or distinguish every legitimate rearrangement.
- 18 / 24 channels retain at least one distance-preserved ion; **14 / 24** retain a full
  product pair (8 / 12 floor, 6 / 12 controls). Twelve also retain a precursor state.
- Balanced electronic reaction proxies exist for 21 channels using all converged states,
  or 12 with the distance-graph filter. No energy was used to fit or score spectral labels.
  Radical calculations use the inferred high-spin unpaired-electron count; alternative spin
  coupling and ground-state multiplicity remain untested.

Independent validation checks **5,764 balanced state pairs**, 4,013 heavy-connectivity graph
isomorphisms, every inferred ion mass against its cached channel (maximum error 5.001e-7 Da),
all 322 unique quantum records and all 297 optimized atom identities/orders and finite energies.
Known heterolysis, alkene bond-order change, methyl radical and impossible closed-shell parity
fixtures pass. Scratch-source Ruff passes; production model/training code and configs have no
changes from this audit.

Continuation artifacts: `output/energetics_joint_states_2026_10_02/` contains protocol and
source identities, matched pilot/full states, full shortlist and channel pack, reconciled
base-peak coverage, production baseline, all geometry jobs, reaction proxies and independent
validation. Source snapshots are text files in the ignored output directory; executable audit
programs remain in `/tmp`. The environment, canonical test spectra/labels and original
checkpoints are untouched. No cluster job, spectral fit or efficacy result exists.

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

The multiple-cut and complete-shortlist continuation above supplies the next feasibility
checkpoint. A full information gate still needs a bounded state-calculation policy, explicit
geometry/missing-state handling, training/calibration molecules and an independent-selector
confirmation design. Passing formal chemistry, convergence and runtime checks does not
establish that the energies improve prediction.
