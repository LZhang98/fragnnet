# MASSter SMARTS templates: CUDA enablement, audit and proposal reach

> **Status:** FINDING · **Updated:** 2026-09-28
> SMARTS templates now run on the CUDA candidate backend that every current MASSter config
> uses. This records the port, an audit of the SMARTS path, how often the trained `c0`
> proposal reaches the template cuts on its own, and what coverage they add. The shared-registry
> matcher defects are fixed, MASSter dropped its ring rules and duplicate templates, and AIMe's
> worked examples are checked. Nothing trained beyond one-epoch local smokes. Arm `t1` is
> prepared, not submitted. Uncommitted.

## Answer

1. **Enabled on CUDA, exact.** The CUDA builder now yields the CPU backend's candidate fields
   exactly, SMARTS rows included. With SMARTS off, every changed path is bit-identical, so `c0`
   and every existing config are unaffected. Arm `t1` is `c0` plus
   `masster_smarts_templates: true`
   ([config](../../../config/masster/mh_ft_smarts_2026_09_27/t1_smarts_templates.yml)).
2. **What SMARTS can change under `c0`.** `c0` runs `proposal_selection_only` with mask-order
   selection and no base-probability feature. There the mechanism head trains through the
   proposal loss only, and nothing reads it on the way to the predicted spectrum. Turning SMARTS
   on therefore does two things:
   - it forces each template's cut mask into the pool;
   - it adds each template side that is disconnected in the precursor as a new fragment.
3. **The proposal already holds most neutral-loss cuts, and almost no ring-rule cuts** (§3).
   - Of the 2-3-cut neutral-loss templates, 71% are in `c0`'s pool without forcing.
   - Of the 4-cut ring rules, 13% are; 78% rank past the 4,096-mask enumeration budget.
4. **Only the neutral-loss products add coverage** (§3.1). Measured as target intensity no
   pool fragment covers:
   - neutral-loss products add 0.196% over random fragments of the same size (95% CI
     0.089-0.333);
   - forcing a ring cut adds 1.3 new fragments that cover no more than random (−0.001%, CI
     −0.006 to 0.005), and ring byproducts cover less than random;
   - forcing a duplicate template's cut adds nothing either (+0.005%, CI −0.010 to 0.029).
5. **The mechanism head scored every retained mask** (about 1,420 per spectrum on `c0`). It
   now scores only masks that carry a template. The values are the same; single-mechanism masks
   are exactly 0.
6. **The 2026-09-23 matcher defects are fixed in the shared registry** (§4).
7. **MASSter dropped the ring rules, `aryl_sulfonate_so3` and duplicate templates** (§5), as
   §3.1 supports. That left 14 rules and 6.1% of `[M+H]+`/FT spectra with an eligible
   template (it was 9.9%).
8. **Eight new MASSter rules from the documented lists pass a per-class gate** (§7). Five
   are clear and three weak. The registry is now 22 rules, and 8.0% of `[M+H]+`/FT spectra
   carry a template.
9. **AIMe's worked examples need no rule, except one** (§6). MASSter's plain cuts reach 14 of
   the 15 annotated fragment masses. The hydroxamic-acid NH rule was initially rejected in
   §6 and later admitted with weak evidence in §7.

## 1. What changed

- **Ranking kernel** (`src/fragnnet/frag/masster_cuda_selection.cu`, `.py`).
  - `rank_batch_cuda(..., extra_forced_bond_sets=...)` places template cuts after the empty and
    single-cut masks.
  - It drops learned duplicates of them, mirroring the CPU `_merge_forced_masks`.
  - The metadata width grows from 13 to 15 columns; the host packs the masks as bit words.
- **Candidate builder** (`src/fragnnet/frag/masster_cuda_full_candidates.py`).
  - `_template_forced_masks` reproduces the CPU forced order.
  - `_append_smarts_mechanisms` inserts each view's SMARTS mechanisms, components, atoms,
    boundaries and new unique fragments after its ordinary rows, in the CPU layout. Earlier
    views' index values are shifted accordingly.
  - A connected template side is exactly an ordinary component of its forced mask, because the
    template cuts every crossing bond and no internal one. Its unique index is read on the
    device, sync-free. A disconnected side is new, deduplicated per view in template order.
  - Grouped views must share template cuts; otherwise the builder raises the CPU message.
- **Model** (`src/fragnnet/model/masster_model.py`).
  - The CUDA guard no longer refuses SMARTS.
  - `_mechanism_log_probabilities` scores only contested masks.
- **Rule set** (`src/fragnnet/frag/masster_templates.py`, §5).
  - The ring rules and `aryl_sulfonate_so3` are out of the MASSter allowlist.
  - `build_masster_templates` skips duplicate templates.
  - `_is_connected` moved here from the CUDA builder, which now imports it.
- **Config.** The `template.yml` comments for `masster_candidate_backend` and
  `masster_smarts_templates` now match the behavior. The arm config is new.
- **Still refusing SMARTS:** `fragment_selection: marginal`, `connected_fragment_beam` and the
  benchmark-only CUDA topology prototype. `c0` uses none of them.

## 2. Validation

- **Tests.**
  - `tests/test_masster_cuda_selection.py`: extra forced masks match the CPU forced merge,
    including learned duplicates, multi-word masks and single-bond extras without forced
    singles. It also covers input validation and word packing.
  - `tests/test_masster_cuda_full_candidates.py`: on six real molecules, CPU and CUDA fields are
    identical and gradients agree, at caps 2, 8 and 256, singles on and off, grouped and not.
    - Real templates supply disconnected products. Synthetic partitions cover the other
      paths: a single-bond cut already in the forced prefix, a multi-cut mask with two
      connected sides, and a disconnected side that becomes a new fragment.
    - A CPU-backend and a CUDA-backend model with SMARTS give the same predictions and
      parameter gradients.
    - Grouped views with different template cuts raise.
  - `tests/test_masster_smarts.py`: the restricted mechanism head equals brute-force scoring of
    every mechanism.
  - The full suite passes with CUDA available: 4,634 passed, 1 skipped, 2 xfailed, after the
    registry fixes, the MASSter drop and the gated additions.
- **Scratch probes.**
  - Kernel against the CPU forced merge: 200 random trials, no mismatch.
  - CPU against CUDA candidates: 60 random batches of real NIST20 molecules, including grouped
    views, caps 4-256 and budgets up to 4,096. Every field is identical; gradients agree within
    fp32 summation order.
- **Real-data smokes** (`c0`, 2,000 spectra per split, one epoch, batch 64, RTX 2070 SUPER).
  - All exit 0: CPU backend with SMARTS, then CUDA with SMARTS on and off.
  - On against off: 15 s against 12 s per training epoch. The first epoch includes template
    matching, which is cached per SMILES afterwards.
  - Val candidate recall: 0.8644 against 0.8639.
  - Val cosine (0.244 against 0.273 after 32 steps) is not read.
- **Builder cost** (192 `[M+H]+`/FT molecules, `c0` budgets, CUDA). 599 ms against 574 ms,
  of which SMARTS assembly is 24 ms, mostly the interleave copying the atom arrays. The
  replaced all-mask mechanism head would have run on 579,092 mechanisms for 27 templates.

## 3. How often the trained proposal reaches template cuts

Probe: `scripts/debug/debug_masster_smarts_proposal_reach.py` on the `c0` checkpoint
(`vcpho5qm`), val, in eval mode with SMARTS off, which is `c0`'s own pool. It covers the
5,061 val spectra whose molecule has an eligible `[M+H]+` template: 6,804 (spectrum,
template) pairs, with output under `output/smarts_reach_2026_09_27/`.

MASSter does not sample the proposal. It enumerates cut masks in exact Bernoulli order and
keeps a prefix until 256 unique fragments. "Reached" means the cut is in that retained pool.

| Template kind | Pairs | In pool | Within the 4,096 budget | log10 pool share (25/50/75%) | nats below the MAP mask (median) | Product explains ≥1% |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Neutral-loss (2-3 cuts) | 3,763 | 71% | 74% | −6.0 / −4.1 / −2.5 | 6.9 | 7.1% |
| Ring rules (4 cuts) | 2,879 | 13% | 22% | −10.8 / −8.6 / −6.6 | 16.4 | 2.3% |
| Single cut (`aryl_sulfonate_so3`) | 162 | 100% (forced) | 83% | −3.7 / −2.0 / −1.2 | 3.0 | 0% |

- **Pool share** is the cut's probability over the pool's total mask probability, i.e. how
  often a draw from the proposal restricted to the pool would be this cut. A template-only
  fragment's proposal probability is at most this, since it also multiplies by a mechanism and
  a charge probability.
- **By rule.** Anthraquinone (both rules), xanthone, furanocoumarin, morphinan arm loss and
  phenoxazine cuts are never in the pool. Halogenated quinoline is in it 2% of the time.
  `aryl_sulfonamide_so2` (83%, median rank 55), `diaryl_ketone_co_loss` (74%) and
  `n_nitramine_n2o` (93%) mostly are.
- **Where the product matters.** 335 pairs (318 spectra) have a template-only side that
  mass-matches at least 1% of the target intensity.
  - For the neutral-loss rules with real signal, the cut is already pooled: methyl benzoate,
    sulfonamide, haloalkyl C2, nitramine and diaryl ketone 100%, aryl sulfonate SO2 98%,
    aminobenzothiazole 94%.
  - The ring-rule matches in this group (bis-CH 38, anthraquinone C4 13, morphinan arm 12) are
    mostly outside the pool. They are merged byproducts, which the
    [2026-09-23 audit](../smarts_usage_audit_2026_09_23.md#22-which-side-of-a-template-is-new-and-does-it-explain-peaks)
    places at the chance floor.
- **Consequence.** Forcing matters almost only for ring rules, whose added fragment is a
  chance-floor byproduct. For neutral-loss rules the proposal already reaches the cut, and
  SMARTS contributes the rearranged product that no ordinary mechanism can form.

### 3.1 Coverage the templates add beyond the pool

The same probe asks what each template adds to the pool's coverage. A peak counts as newly
covered when it is within tolerance of a new fragment's model channel and of no channel of
any pool fragment. New fragments get the model's own channels: adduct, H-shift window and
valence bound.

- **Forced cut:** the ordinary pieces of the template's cut that the pool lacks.
- **Template side:** the disconnected side, i.e. the neutral-loss product or the ring
  byproduct.
- **Null:** random connected atom sets of the same sizes, also missing from the pool.

Figures are the mean percentage of the spectrum's intensity. Kinds: duplicate = both sides
connected; ring = 4 cuts; neutral loss = the rest.

| Kind | Pairs | New fragments per template | Pool covers | Forced cut adds | Null | Template side adds | Null |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Neutral loss | 3,309 | 0.07 | 90.4% | 0.000% | 0.000% | 0.19% | 0.000% |
| Ring | 2,879 | 1.30 | 94.3% | 0.010% | 0.010% | 0.000% | 0.02% |
| Duplicate | 616 | 0.41 | 91.4% | 0.05% | 0.04% | — | — |

Differences against the null, resampling molecules (2,000 draws):

| Contrast | Difference | 95% CI | Molecules |
| --- | ---: | --- | ---: |
| Neutral-loss product | +0.196% | 0.089 to 0.333 | 239 |
| Ring forced cut | −0.001% | −0.006 to 0.005 | 170 |
| Ring byproduct | −0.013% | −0.022 to −0.006 | 170 |
| Duplicate forced cut | +0.005% | −0.010 to 0.029 | 49 |

- **Neutral-loss products that add coverage:**
  - `aryl_sulfonate_so2` adds 1.8% on average, and at least 1% on 29% of its templates;
  - `n_nitramine_n2o` 0.43% (18%), `short_haloalkyl_ammonium_c2_transfer` 0.49% (10%) and
    `aryl_sulfonamide_so2` 0.22% (4%).
  - Diaryl ketone, methyl benzoate, diarylvinyl, aminobenzothiazole and the C3 haloalkyl
    rule add nothing on `[M+H]+`/FT. The peaks they mass-match are already covered by pool
    fragments, so the "product explains" column of §3 overstates them.
- **This overturns the ring-rule half of the 2026-09-23 audit's ranking.** That audit found
  the ordinary side of a ring cut explains real intensity, but did not check whether the pool
  already covers it. It does: forcing the cut adds fragments no better than random.

`[M-H]-` rules and spectra are not in `c0`'s scope and were not measured.

## 4. Audit of the SMARTS path

| Item | Status | Evidence |
| --- | --- | --- |
| Template atom and bond order against the model graph | Correct | `Chem.MolFromSmiles(smiles)`, which the PL matches, and the stored standardized `mol` agree on atom and bond order for all 35,025 NIST20 molecules |
| Tautomer drift between the two molecules | Minor, not fixed | Bond types differ for 155 molecules, templates for 3 (phenoxazine). The PL has no stored `mol` in the batch |
| CPU forced-mask deduplication | Correct | Learned masks list bonds in increasing order (85,192 masks checked), so template cuts match their ranked duplicates |
| CPU lookup of existing fragments | Correct | No duplicate atom set among 19,615 unique fragments of 300 template molecules |
| Redundant templates | Harmless, costs a mechanism | 440 of 4,544 templates (9.7%) have both sides connected, so they repeat the ordinary partition. This includes all 181 `aryl_sulfonate_so3`. No template adds two new sides |
| Mechanism head under `c0` | Inert for predictions | Only the proposal loss and diagnostics read it (§Answer 2) |
| Mechanism head cost | Fixed | It scored every retained mask; now only masks with a template (§1) |
| Standalone inference | Correct | `MassterPL.forward` builds templates when the sidecar is missing |
| `mol_params.smiles` | Precondition | Default `True`; with it off, SMARTS fails with a `KeyError` at transfer |
| Rule identity downstream | Not used | It is not an intensity-head input and not exported with the per-peak annotation |
| Sulfoxide rule dead | Fixed (shared registry) | `~[OX1]` accepts the standardizer's `[S+][O-]`; it now matches all 120 NIST20 sulfoxides, where it matched none |
| `uniquify=True` in the anthraquinone rules | Fixed (shared registry) | New per-rule `SmartsFragRule.uniquify`, off for the two rules: 2 and 4 partitions per anthraquinone instead of 1. Other rules keep the default, so the broad `crf12`/`crf13` matches and RDKit's match limit are untouched |
| Wrong comments | Fixed | The quinoline mirror claim (only C6 is excised); the NL H-shift values (0 for SO2, SO3 anion, SO, CO and C2H2; −1 for HCN; +1 for N2O and the SO3 acid); the nitramine structure (O2N–NH–R); the prepass depth (1 on the multi-cut path, 2 on the single-cut path) |
| Effect on FraGNNet | Recorded | On all 35,025 NIST20 molecules only those three rules change, and nothing is removed. Old outputs are an ordered subsequence of the new, so existing node order is kept. Existing DAGs are unaffected; the note is in `preproc_scripts/README.md` |

## 5. Decisions (2026-09-27)

1. **Ring rules: dropped from MASSter.** All eleven compound ring-cut rules are gone. Neither
   the forced cut nor the byproduct adds coverage beyond random fragments (§3.1). This reverses
   the forced-masks option the
   [2026-09-23 audit](../smarts_usage_audit_2026_09_23.md#5-what-would-use-smarts-better-ranked)
   ranked second. FraGNNet's prepass keeps them.
2. **Duplicate templates: dropped entirely.** `build_masster_templates` skips any template
   whose sides are all connected, so it adds neither a forced cut nor a mechanism.
   `aryl_sulfonate_so3`, whose single-bond cut is always a duplicate, left the allowlist.
3. **Rule indices were renumbered** to 1-14, and `MASSTER_NUM_RULES` is now 14. No SMARTS
   checkpoint existed.
4. **Eight gated rules were added** from the documented candidate lists (§7). They are
   MASSter-only and admitted only in the polarity that passed. Indices are now 1-22.
5. **Still open:** whether to submit `t1`. Its read is in the config header: candidate recall
   first, then the 54-63 tail against `c0`, then the template spectra alone.

## 6. AIMe's worked examples and the other rule lists

**Status clarification (2026-09-28):** this section records the initial rejection of the
hydroxamic NH rule. The subsequent §7 gate admitted it with weak evidence and it is present
in the current registry. The 14/15 result is exhaustive mass reachability, not an assertion
of product connectivity or retention under a finite proposal budget. See the
[literature follow-up](../../notes/masster_excision_literature_2026_09_28.md).

The question: do our rules cover the examples in AIMe
([note](../../notes/aime_deepms2reasoner.md))? AIMe builds fragments with three operators:
- linear-bond cleavage;
- ring-bond cleavage;
- a generic excision, which removes an internal substructure as a neutral loss and reconnects
  its boundary atoms.

It annotates 15 fragments:
- Fig. 2c: 4-chloro-2-hydroxybenzohydroxamic acid, `[M+H]+`, 8 ions;
- Fig. 3k: biotin, `[M-H]-`, 7 ions.

Each ion was checked against MASSter's own channels (`c0` settings: valence bound, ±4 H, no
CMF, 10 ppm) over every connected atom subset and every MASSter template side.

- **14 of 15 are reachable by plain cuts.**
  - Biotin's multi-step fragments (166.087, 122.098) are connected atom subsets, because its
    excisions happen inside rings.
  - AIMe's highlighted imide C=O excision, C6H5ClNO+ at 142.005, is reached only by an
    isomeric connected atom set. AIMe's own product atom set, with the new C–N bond, is
    disconnected in the precursor.
- **The miss is C7H6ClO3+ (173.000).** It is an NH excision that moves the N–OH oxygen onto
  the carbonyl carbon. No rule in any list produces it: MASSter, FraGNNet's prepass, or the
  lists below.
  - NIST20 has 50 primary hydroxamic acids (1,283 `[M+H]+`/`[M-H]-` spectra).
  - The standardizer stores them as the hydroximic tautomer `ON=C(O)R`, the same trap that
    killed the sulfoxide rule.
  - The product peak holds 0.06% (`[M+H]+`) and 0.05% (`[M-H]-`) of intensity on average, and
    at least 1% in under 2% of spectra. That is the chance floor, so no rule was added.
- **The other rule lists.**
  - [`smarts.md`](../../reference/smarts.md): the 72-rule PPGB-MS2 survey, whose reaction
    SMARTS also sit in `preproc_scripts/no_dag_exp/03_test_other_smarts.py`. Run single-step
    on both molecules, its 101 parseable reactions reach 6 of the 15 ions.
  - [`rearrangement_smarts.md`](../../reference/rearrangement_smarts.md): 7 validated SMIRKS
    (McLafferty, RDA, retro-ene, Cope, Claisen), not integrated because cuts cover them.
  - The ESI-CID backlog in [`masster_smarts_rules.md`](../../reference/masster_smarts_rules.md):
    about 22 unimplemented literature candidates.
  - None has a hydroxamic or Lossen rule.
- **Consequence.** A bond-cut model reaches any excision that happens inside a ring. What
  it cannot reach is acyclic excision, which is what MASSter's neutral-loss rules encode case
  by case. AIMe's generic operator is the systematic form of those rules.

## 7. New rules from the documented lists

At the user's direction, rules were added for the cases plain cuts cannot reach. The
candidates came from:
- the ESI-CID list in [`masster_smarts_rules.md`](../../reference/masster_smarts_rules.md);
- the 09-23 audit's sulfonamide variants;
- AIMe's hydroxamic example.

The ChemFrag triage's only lead, the Wagner–Meerwein migration, still lacks a post-migration
cut and was not encoded. Probe: `scripts/debug/debug_masster_smarts_rule_gate.py`, whose output is in
`output/smarts_reach_2026_09_27/rule_gate.pkl`.

**Step 1, census** on standardized molecules, NIST20 spectra of the rule's polarity:

| Candidate | NIST20 molecules | Spectra | NIST23 molecules |
| --- | ---: | ---: | ---: |
| primary sulfonamide (SO2 loss / O transfer) | 185 | 3,389 / 2,542 | 287 |
| tertiary sulfonamide SO2 | 191 | 3,320 | 550 |
| phenylalanine O transfer | 68 | 1,877 | 94 |
| hydroxamic NH excision | 50 | 756 | 67 |
| N-acyl sulfonamide O transfer | 32 | 158 | 51 |
| N-methylanilinoethanol | 18 | 32 | 34 |
| sulfonylurea, PFAS sulfonate | 16 each | 106 / 101 | 17 / 19 |
| benzyl phenyl sulfone | 10 | 134 | 17 |
| diacylhydrazine, pirimicarb, proline benzyl ester, dichlorvos, methacrylamide | 1-4 | 0-78 | 1-5 |
| nitrobenzyl ester, prazole sulfone, pyrimidylsulfanyl benzamide, benzaldimine, dibenzylaniline | 0 | 0 | 0-4 |

**Step 2, gate.**
- **Peak coverage:** for each eligible spectrum, the target intensity the rule's
  disconnected product explains through MASSter's own channels (valence bound, ±4 H, no
  CMF) at the spectrum's instrument tolerance. Only peaks that no connected atom set of the
  same heavy composition can reach within 4 H count; that set is found by exhaustive
  enumeration under a composition bound.
- **Null:** random disconnected atom sets of the same size, scored identically.
- **Statistics:** molecule-bootstrap CIs over 1,000 resamples.

Results, in % of spectrum intensity:

| Rule | Kind | Mode | Molecules | Unique | Null | Difference (95% CI) | Spectra ≥1% |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| `primary_aryl_sulfonamide_so2` | new | `[M-H]-` | 67 | 6.04 | 0.13 | +6.39 (3.71, 9.59) | 24.5% |
| `primary_aryl_sulfonamide_so2` | not admitted | `[M+H]+` | 175 | 0.20 | 0.27 | −0.11 (−0.67, 0.27) | 2.1% |
| `primary_aryl_sulfonamide_o_transfer` | new | `[M+H]+` | 175 | 2.53 | 0.07 | +2.61 (1.85, 3.45) | 22.2% |
| `n_acyl_aryl_sulfonamide_carbonyl_o_transfer` | new | `[M-H]-` | 10 | 6.64 | 0.00 | +5.69 (0.79, 12.85) | 28.6% |
| `diacylhydrazine_carboxylate_o_transfer` | new | `[M-H]-` | 3 | 43.1 | 0.00 | +49.6 (0.0, 78.7) | 59.5% |
| `pfas_sulfonate_f_transfer` | new | `[M-H]-` | 10 | 2.54 | 0.00 | +1.43 (0.00, 3.38) | 22.0% |
| `phenylalanine_carboxyl_o_transfer` | new, weak | `[M+H]+` | 66 | 0.06 | 0.00 | +0.15 (0.00, 0.36) | 1.1% |
| `tertiary_aryl_sulfonamide_so2` | new, weak | `[M+H]+` / `[M-H]-` | 174 / 57 | 0.11 / 0.06 | 0 / 0 | +0.11 / +0.07 | 0.8% / 1.4% |
| `hydroxamic_acid_nh_excision` | new, weak | `[M+H]+` | 47 | 0.09 | 0.05 | +0.04 (0.01, 0.08) | 2.5% |
| `aryl_sulfonamide_so2` | existing | `[M+H]+` / `[M-H]-` | 446 / 237 | 0.52 / 3.41 | 0 / 0 | +0.29 / +3.75 | 7.1% / 18.1% |
| `aryl_sulfonate_so2` | existing | `[M+H]+` / `[M-H]-` | 86 / 121 | 1.90 / 6.75 | 0.22 / 0.07 | +2.06 / +6.40 | 22.7% / 34.6% |
| `n_nitramine_n2o` | existing | `[M+H]+` / `[M-H]-` | 10 / 5 | 0.34 / 3.89 | 0 / 0 | +0.20 / +6.98 | 10.6% / 9.7% |
| `short_haloalkyl_ammonium_c2_transfer` | existing | `[M+H]+` | 29 | 0.33 | 0.38 | −0.01 | 2.1% |
| `sulfoxide_so_loss` | existing | `[M+H]+` | 90 | 0.03 | 0.01 | −0.05 | 0.7% |
| diaryl ketone, methyl benzoate, diarylvinyl, aminobenzothiazole, N-phenylbenzamide, haloalkyl C3 | existing | all | — | 0.00 | 0.00 | 0.00 | 0% |

- **Why CO-type losses score zero.** A loss made of common elements (CO, CO2, HCN, C2H2)
  always has a formula twin among connected atom sets: drop a different C and O, and the
  rest stays connected. Such a rule adds no m/z the model could not already produce, only a
  more plausible explanation of it. The sulfonyl and nitramine losses are unique because
  they remove a bridging heteroatom.
  - The earlier `c0`-pool measure (§3.1) counts formulas missing from the actual pool.
    There the C2 haloalkyl rule added +0.49%: its formula twin exists but is not in `c0`'s
    256-fragment pool.
- **Registry changes** (MASSter-only, `masster_templates.py`, restricted to the passing
  polarity).
  - Two compound-fragment rules and six neutral-loss rules added.
  - Rule indices are now 1-22.
  - New tests in `tests/test_masster_smarts.py` check each rule on a standardized positive
    (product composition, polarity gating) and on close negatives.
- **Limits.**
  - The gate is formula-level and model-free: it bounds what the rule could add, not what a
    trained model gains.
  - Several classes have 3-10 molecules, so their CIs are wide.
  - The diacylhydrazine and PFAS lower bounds sit at 0 because resampling can draw only the
    zero-scoring molecules.

## 8. Method and limits

- §3 ran on the pre-drop registry (26 rules, duplicates kept). Rerunning the probe now skips
  the ring and duplicate templates; its saved table is in `output/smarts_reach_2026_09_27/`.
- The reach probe uses a proposal trained without templates. With SMARTS on, the proposal loss
  would push up the cuts whose products explain peaks, so §3 describes the starting point, not
  the trained arm.
- "Product explains" is a mass match (10 ppm, 200 m/z floor, H shifts −4..+4). It is an upper
  bound that includes chance matches and intensity other fragments already explain.
- The smoke and benchmark ran on an RTX 2070 SUPER at small batches; H100 costs were not
  measured.
- The kernel and candidate parity scripts were scratch files. The committed tests cover the
  same paths at smaller scale.
