# ChemFrag rearrangement rule triage

> **Status:** FINDING · **Updated:** 2026-09-28 · Symbolic and substructure probes complete; no
> rule implemented. The 2026-09-28 coverage check closes the set: **none of the 32 transforms
> adds a fragment that plain cuts cannot, and one that explains peaks beyond chance**, including
> the Wagner–Meerwein Candidate A. See
> [Unique-coverage check](#unique-coverage-check-2026-09-28).

## Scope and conclusion

A 32-transform ChemFrag rearrangement set (reaction SMARTS, `>>`) was supplied for triage
against the three routes named in the request: fusion into `SmartsFragRule` /
`NeutralLossRule`, use by MASSter, and a charged-species rearrangement stage that does
not exist today. The set corresponds to the ChemFrag collection already listed as a
comparison priority in the [SMARTS set and literature review](masster_smarts_review_2026_09_13.md#other-rule-collections-to-compare)
(Schüler et al., [J. Mass Spectrom. 2025](https://doi.org/10.1002/jms.5129); 44 cleavage plus
16 rearrangement rules, several supplied here as forward/reverse pairs).

**Two measurements decide almost the whole set.** Twenty-six of the 32 transforms leave the
heavy-atom connectivity unchanged, and 29 of the 32 cannot match a standardized neutral
precursor at all. The intersection of "changes the heavy-atom skeleton" and "matchable on a
neutral precursor" is **empty**, so no transform in this set converts mechanically into a
registry rule.

Only the Wagner–Meerwein alkyl migrations (#20, #21) are worth further work, as a MASSter
rule whose cation guard is replaced by a cation-precursor guard and whose incremental coverage
is measured first. Deleting the charge constraints does not make that pair too loose; it does
make the rest of the set `crf12`-broad, but those are already excluded on gate 1. This is the same steroid methyl-shift lead the previous review already extracted from
the ChemFrag article; the present triage adds that it is the *only* member of the set with
both a new heavy-atom assembly and non-trivial scaffold occurrence.

## Method

Two probes, both run on the `FRAGNNET-GPU` interpreter over `data/proc/nist23/mol_df.pkl`
(52,787 standardized neutral molecules).

1. **Skeleton-change probe.** Each transform was parsed with
   `AllChem.ReactionFromSmarts`. For the reactant and product templates, the set of bonds
   between mapped heavy atoms was collected and keyed by atom-map pair, restricted to map
   numbers that are heavy on both sides. A non-empty symmetric difference means the transform
   forms or breaks a heavy-atom bond. This is exactly admission gate 1 in the
   [SMARTS rule catalog](../../reference/masster_smarts_rules.md#scope-and-admission-rule):
   a rule is useful only when it creates a new heavy-atom assembly.
2. **Precursor-matchability probe.** The reactant half of each transform was compiled with
   `MolFromSmarts` and matched against every molecule, with and without `Chem.AddHs`.
   The prepass runs at depth 0 on the neutral precursor
   ([multi_cut_bfs.pyx:851](../../../src/fragnnet/frag/multi_cut_bfs.pyx#L851)), so a template
   that cannot match there cannot be used as written.

3. **Charge-relaxation probe.** Charge primitives (`[+-]\d*`) and, separately, open-valence
   primitives (`v\d+`) were deleted from each reactant template's atom expressions, leaving
   atom maps intact, and the relaxed templates were re-matched with `GetSubstructMatches` to
   record both the molecules hit and the number of matches per molecule. The same measurement
   was run over the 24 shared registry rules to give "too loose" a calibrated scale.

Coarse neutral-scaffold occurrence proxies were then written by hand for the six
skeleton-changing transforms, to estimate how often the underlying chemistry could apply if
the template were rewritten for a neutral precursor. These proxies are upper bounds on
occurrence and carry no evidence about product formation.

## Probe results

**Skeleton change: 6 of 32.** Transforms #13, #14, #15 and #16 form one bond and break none
(cyclization onto a heteroatom). Transforms #20 and #21 both form and break a bond — an alkyl
group migrates from one ring-fusion carbon to the adjacent cation centre. The other 26 change
only hydrogen positions, bond orders, formal charges, or the partition between an ion and its
neutral partner.

**Precursor matchability: 3 of 32, none of them skeleton-changing.**

| Transform | Matches (52,787 mols) | Why the rest fail |
|---|---:|---|
| #10 phenol → cyclohexa-2,4-dienone | 6,698 (12.69%), needs `AddHs` | — |
| #22 allylic 1,3-H shift | 1,045 (1.98%), needs `AddHs` | — |
| #23 1,5-H shift (same reactant template) | 1,045 (1.98%), needs `AddHs` | — |
| all other 29 | 0 (0.00%) | require a formal charge, a divalent carbene (`v2`), or a trivalent radical carbon (`v3`) |

The zero counts are structural, not incidental: the templates describe charged or
open-valence intermediates that exist only inside a fragmentation tree. This is the same
failure mode recorded for `nl_rule_sulfoxide_so_loss`, which matches 0 of 52,787 molecules
because the standardizer writes `[S+][O-]`.

**Neutral-scaffold occurrence proxies for the six skeleton-changing transforms:**

| Transform | Proxy SMARTS | Molecules | Share |
|---|---|---:|---:|
| #13 benzylic X, ortho vinyl | `[#6,#7,#8;!$([#6]=O)]-[CX4H2]-c1ccccc1-[CX3]=[CX3]` | 44 | 0.08% |
| #14 benzylic X, ortho alkynyl | `[#6,#7,#8]-[CX4H2]-c1ccccc1-[CX2]#[CX2]` | 8 | 0.02% |
| #15 ortho alkoxymethyl aryl carbonyl | `[CX4]-[OX2]-[CX4H2]-c1ccccc1-[CX3]=[OX1]` | 1 | 0.00% |
| #16 thioether/thiol 1,4 to carbonyl | `[SX2;H1,H0]-[#6]-[#6]-[#6]=[OX1]` | 412 | 0.78% |
| #20/#21 angular methyl at a ring fusion | `[CH3]-[CX4;R2]` | 3,344 | 6.33% |
| #20/#21 angular methyl on a 6-5 or 6-6 fusion | `[CH3]-[CX4;R2]([#6;R])([#6;R])[#6;R]` | 3,065 | 5.81% |

## Removing the charge constraints

Charge and open-valence primitives were deleted from every reactant template
(`[#6+:2]` → `[#6:2]`, `[#6;A;v2;+0:2]` → `[#6;A:2]`) and the relaxed templates re-matched.
"Too loose" is judged against the registry's own admitted breadth, measured the same way on
the same molecules:

| Reference set | Molecules matched | Matches per matching molecule |
|---|---|---|
| 21 scaffold rules in the shared registry | 0.01%–1.51%, median 0.40% | 1.0–1.9 |
| `crf12_0`, excluded from MASSter as too broad | 77.51% | 21.5 |
| `crf12_1`, `crf13_2`, likewise excluded | 21.95%, 19.93% | 2.9, 1.4 |

Against that scale the answer is rule-dependent, and it splits along the same line as the
rest of the triage.

**Too loose — the skeleton-preserving majority.** Deleting charge from the resonance, H-shift
and ion–neutral-complex transforms produces patterns at or beyond `crf12` breadth, because
the charge was the only thing restricting them. #12 (acylium → `C≡O⁺`) relaxes to "any
carbonyl", 57.99% of NIST23. #6 reaches 48.07% at 14.1 matches per molecule, #7 36.25%, #11
36.10%, #18 23.04%. The ion–neutral-complex rules are worse on candidate load than breadth:
#27 matches only 10.24% of molecules but yields **80.6 matches per molecule**, well past
`crf12_0`, because the `.` makes the two halves match independently anywhere in the graph.
None of these passes gate 1 regardless, so the looseness question is moot for them.

**Not too loose — the migration rules.** #20 and #21 relax to 2.34% and 3.25% of molecules at
1.1 and 1.6 matches per molecule. The fused-ring topology, not the charge, was carrying the
selectivity. That is roughly twice the current admitted ceiling but the same order, and the
per-molecule candidate load is at the low end of the registry.

They tighten further if the deleted cation is replaced by a **cation precursor** rather than
simply dropped, which is also the chemically correct guard: the ChemFrag migration acts on a
dehydrated intermediate, so requiring a hydroxyl at the migration terminus encodes the step
that generates the cation.

| Variant | Molecules | Share | Matches/mol |
|---|---:|---:|---:|
| #20 charge deleted | 1,235 | 2.34% | 1.1 |
| #20 + angular methyl | 1,104 | 2.09% | 1.1 |
| #20 + OH at the migration terminus | 219 | 0.41% | 1.0 |
| #21 charge deleted | 1,714 | 3.25% | 1.6 |
| #21 + angular methyl | 1,559 | 2.95% | 1.3 |
| #21 + OH at the migration terminus | 120 | 0.23% | 1.1 |

At 0.41% and 0.23% with ~1.0 matches per molecule, these sit inside the admitted band, close
to the registry median. Breadth is therefore not the obstacle for Candidate A; the missing
post-migration cut specification and the absence of scaffold-specific evidence are.

## Per-transform disposition

`Skel` is the skeleton-change probe result. `Match` is whether the reactant template matches
any neutral precursor.

| # | Transformation | Skel | Match | Disposition |
|---:|---|:---:|:---:|---|
| 1 | Ketenimine anion → nitrile carbanion | no | no | Gate 1 — resonance/bond-order only |
| 2 | Reverse of #1 | no | no | Gate 1 |
| 3 | α-Carbanion ketone → enolate | no | no | Gate 1 |
| 4 | Reverse of #3 | no | no | Gate 1 |
| 5 | Carbene 1,2-H shift → alkene | no | no | Gate 1; also needs a `v2` carbene |
| 6 | 1,2-Biscarbene → alkyne | no | no | Gate 1; needs two `v2` carbenes |
| 7 | Carbene + nitrenium → nitrilium | no | no | Gate 1 |
| 8 | C/N radical cation → iminium | no | no | Gate 1; needs a `v3` radical carbon |
| 9 | Azomethine ylide → neutral imine | no | no | Gate 1 |
| 10 | Phenol → cyclohexa-2,4-dienone | no | **yes** | Gate 1 — H position and bond orders only. See note below |
| 11 | Hydroxycarbene → aldehyde | no | no | Gate 1 |
| 12 | Acylium → `C≡O⁺` | no | no | Gate 1 — pure charge migration |
| 13 | Benzylic cation cyclizes onto a benzylic heteroatom | **yes** | no | Gate 3 — 0.08% occurrence; not worth a rule |
| 14 | Same from an alkynyl cation | **yes** | no | Gate 3 — 0.02% occurrence |
| 15 | Acylium cyclizes onto an ortho alkoxymethyl oxygen | **yes** | no | Gate 3 — one molecule in NIST23 |
| 16 | Thioether S attacks a protonated carbonyl → cyclic sulfonium | **yes** | no | Backlog. 0.78% occurrence; needs its own evidence and a post-cyclization cut |
| 17 | 1,2-H shift onto a nitrenium N⁺ | no | no | Gate 1 |
| 18 | 1,2-H shift onto an oxenium O⁺ | no | no | Gate 1 |
| 19 | 1,3-H shift onto an aminium N⁺ | no | no | Gate 1 |
| 20 | Wagner–Meerwein alkyl migration, 6-5 fused cation | **yes** | no | **Candidate A** — see below |
| 21 | Wagner–Meerwein alkyl migration, 6-6 fused cation | **yes** | no | **Candidate A** |
| 22 | Allylic 1,3-H shift in a 1,3-diene | no | **yes** | Gate 1, and already removed from the project set as thermally forbidden |
| 23 | 1,5-H shift in a 1,3-diene | no | **yes** | Gate 1 — isomerization only |
| 24 | Iminium + base → enamine + protonated base | no | no | Gate 1; ion–neutral-complex proton transfer, already in the H-transfer channel |
| 25 | 1,2-H shift, carbocation → iminium | no | no | Gate 1 |
| 26 | 1,2-H shift, carbocation → nitrilium | no | no | Gate 1 |
| 27 | Carbocation + base → alkene + protonated base | no | no | Gate 1; H-transfer channel |
| 28 | Vinyl cation + base → alkyne + protonated base | no | no | Gate 1; H-transfer channel |
| 29 | Carbocation + aldehyde → alkene + protonated carbonyl | no | no | Gate 1; H-transfer channel |
| 30 | Vinyl cation + aldehyde → alkyne + protonated carbonyl | no | no | Gate 1; H-transfer channel |
| 31 | Carbocation + ketene → alkene + acylium | no | no | Gate 1; H-transfer channel |
| 32 | Vinyl cation + ketene → alkyne + acylium | no | no | Gate 1; H-transfer channel |

Transform #22 is the allylic 1,3-H shift that the
[rearrangement review](../../reference/rearrangement_smarts.md#rules-removed) already removed
as a thermally forbidden concerted sigmatropic shift. ChemFrag's inclusion of it is not by
itself a reason to reinstate it; a stepwise or ion-catalysed gas-phase pathway would need its
own evidence.

Transform #10 matches 12.69% of NIST23 and is chemically real, but it moves one hydrogen and
redistributes bond orders. It fails gate 1 as a partition rule. Its only value would be as a
dearomatized isomer fed back into fragment enumeration, which is the isomer-prepass layer
discussed below.

## Candidate A: Wagner–Meerwein alkyl migration (#20, #21)

This is the one transform family that satisfies gate 1 for a reason that survives inspection.
The migrating group ends up bonded to a carbon it is not bonded to in the precursor, so a
product retaining the migrating alkyl together with the distal ring piece — and not the
intervening carbons — is disconnected in the precursor graph. MASSter's ordinary
connected-component mechanism cannot express that partition under any cut mask, which is
precisely the condition for admitting a rule.

It is also the family the previous review flagged from the same article, noting that
"migration alone preserves mass; only subsequent cleavage can demonstrate new support."
That remains the blocking design question: a `SmartsFragRule` must specify both the
post-migration cut bonds and the `merge_outer_map_nums` assembly, and the ChemFrag transform
specifies neither.

Three things must happen before any implementation:

1. **Rewrite for a neutral precursor.** The supplied templates require `[#6+]`. A registry
   rule must instead guard the neutral scaffold — a 6-5 or 6-6 fused ring system with an
   angular alkyl group — the way `morphinan_amine_arm_loss` and the other compound-fragment
   rules do.
2. **Replace the cation guard, do not delete it.** Simply deleting the charge gives 2.34% and
   3.25% of NIST23 at ~1.1 matches per molecule — already within an order of the admitted band,
   and much tighter than the hand-written `[CH3]-[CX4;R2]` proxy above suggests. Requiring a
   hydroxyl at the migration terminus, which encodes the water loss that generates the cation,
   brings them to 0.41% and 0.23% at ~1.0 matches per molecule, inside the registry's normal
   range. Gate 3 is satisfiable here.
3. **Measure before implementing.** Follow the review's
   [coverage check](masster_smarts_review_2026_09_13.md#recommended-coverage-check):
   incremental intensity per rule on fixed filtered validation IDs, then again under the
   production candidate budget. This matters more than usual here: an earlier bond-independence
   probe (2026-08-23, not written up under `docs/`) found that buying additional coverage cost
   cosine on the trained model while weighted recall was already 0.895. That result should be
   re-confirmed or recorded before it is used to justify a decision.

## The two routes that are closed

**Fusion into the shared FraGNNet `FRAG_RULES` / `NL_RULES`.** None of the 32 qualifies.
FraGNNet DAG nodes are atom masks of the precursor; the 26 skeleton-preserving transforms
produce no new mask, and the 6 skeleton-changing ones cannot match the neutral precursor and
do not specify a cut. The
[integration decision](../../reference/rearrangement_smarts.md#coverage-vs-current-bfs-integration-decision)
already reached the same conclusion for the Cope and Claisen isomerizations.

**A charged-species rearrangement stage.** This is where 26 of the 32 would have to live, and
it is a larger change than it first appears. It requires isomer-tracking nodes — a fragment
identified by a bond graph rather than an atom mask — because the whole point of these
transforms is that two nodes with identical atom sets fragment differently. That layer is the
"isomer prepass" the rearrangement review scoped and declined, and nothing in the current
evidence argues for reopening it: the transforms that would populate it are precisely the ones
that add no candidate formula, and the project's own measurements say candidate coverage is
not the binding constraint. Build it only if a residual analysis shows missed intensity that
is specifically attributable to isomer-dependent fragmentation.

## Validation and limitations

- Both probes were executed on the configured interpreter against the full 52,787-molecule
  NIST23 table. Scripts were run from the session scratchpad and not added to the repository.
- The skeleton-change probe compares SMARTS templates, not sanitized products. It answers
  "does this transform form or break a heavy-atom bond", which is the admission gate, and
  makes no claim about whether RDKit can construct the product.
- The charge-relaxation probe deletes primitives mechanically; it is not the same as writing
  the neutral precursor form by hand. Six relaxed templates (#1, #15, #16, #24, #31, #32) match
  nothing because deletion leaves an impossible valence — a protonated carbonyl becomes a
  divalent `=O` bearing an explicit H, for example. Those zeros are artifacts of the method and
  are not evidence that the rules are narrow.
- Scaffold-occurrence proxies were written for this triage and are not validated SMARTS. They
  bound how often a rewritten rule could match; they say nothing about whether the rearranged
  product forms, or whether the resulting mass is already covered by an ordinary candidate.
- No rule, config, model, or test was changed. No coverage evaluation, training run, or test
  suite execution was performed. ChemFrag's own supporting tables were not consulted; the
  triage is of the 32 transforms as supplied.

## Unique-coverage check (2026-09-28)

The same 32 transforms were resubmitted, and the question became whether any covers a case
plain cuts cannot reach. The gate is the one used for the MASSter rules admitted on
2026-09-27
([finding §7](masster_smarts_cuda_audit_2026_09_27.md#7-new-rules-from-the-documented-lists)).

A transform can only add a candidate by forming or breaking a heavy-atom bond, because MASSter
and FraGNNet represent a fragment as a precursor atom set plus an H shift.

**Re-verified on NIST20.** 26 transforms change no heavy-atom bond, and 3 match a
standardized precursor (#10, #22, #23), none of them skeleton-changing. That is the triage's
split, reproduced.

**Skeleton-changing six.**
- **Method.**
  - Each was rewritten for a neutral precursor, and the bond change applied to every match.
  - Fragments one or two cuts away (within four bonds of the site) in the rearranged graph,
    and disconnected in the precursor, were collected.
  - A fragment is **unique** if no connected atom set of the precursor has the same heavy-atom
    composition within 4 H.
  - Unique fragments were then scored on the `[M+H]+` spectra with MASSter's channels, against
    random disconnected sets of the same size.

| Transform | NIST20 molecules | Unique fragments | Spectral gate |
| --- | ---: | --- | --- |
| #13 benzylic X onto ortho vinyl | 29 | 2 of 18, in 1 match | 5 molecules, 0.00% |
| #14 same, ortho alkynyl | 8 | 1 of 24 | no scorable spectra |
| #15 alkoxymethyl O onto acylium | 0 | — | — |
| #16 thioether S onto carbonyl | 173 | 126 of 553, in 57 of 100 matches | 148 molecules, 1,911 spectra: −0.036% vs random (95% CI −0.081 to −0.003) |
| #20 Wagner–Meerwein, 6-5 | 1,069 | 0 of 263: all have a single-atom swap twin | — |
| #21 Wagner–Meerwein, 6-6 | 1,382 | 0 of 426 | — |

- **#16 is the only transform with real unique fragments, and they explain peaks below
  chance.**
- **Wagner–Meerwein can add no m/z.** Swapping the migrated methyl for the carbon it left
  gives a same-formula connected fragment for 68% (#20) and 95% (#21) of the new fragments.
  Some single-atom swap does it for all of them.
- **Decision.** Add none of the 32. Candidate A is closed.
- **Limits.**
  - Two-cut neighbourhood only.
  - #20/#21 were checked on the first 100 matching molecules.
  - The general twin search was capped (a capped fragment counts as having a twin), which
    errs against the transform.
  - Scripts ran from the session scratchpad.
