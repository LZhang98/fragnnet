# MASSter SMARTS templates: review, CUDA path and gates

> **Status:** FINDING · **Updated:** 2026-10-18
> Five SMARTS findings, oldest first: the rule and literature review, the CUDA port and audit, outlier reach on NIST20, the Enveda-180 per-class gate, and the pathway gates (CLOSED NULL).

Merged 2026-10-18 from 5 findings on one topic. Each part below is the original
document unchanged except for heading levels; its own status line keeps its original date.
Former file names are listed so old links and `git log -- <former path>` can be traced.

| Part | Updated | Verdict | Former file |
| --- | --- | --- | --- |
| [MASSter SMARTS set and literature review](#masster-smarts-set-and-literature-review) | 2026-09-13 | Registry and tests checked; literature review; coverage gains unvalidated | `masster_smarts_review_2026_09_13.md` |
| [MASSter SMARTS templates: CUDA enablement, audit and proposal reach](#masster-smarts-templates-cuda-enablement-audit-and-proposal-reach) | 2026-09-28 | Templates run on CUDA, OFF by default; SMARTS vs CUDA-template mismatch is a hard error | `masster_smarts_cuda_audit_2026_09_27.md` |
| [MASSter SMARTS rules and outlier spectra](#masster-smarts-rules-and-outlier-spectra) | 2026-09-29 | Rules matter only on 37 outlier val spectra; goal = raise the lower bound | `masster_smarts_outlier_reach_2026_09_29.md` |
| [MASSter SMARTS rules on Enveda-180: per-class gate](#masster-smarts-rules-on-enveda-180-per-class-gate) | 2026-09-29 | Enveda-180: only S=O rules pass, 2-10x weaker than NIST20; no support for the 8 disabled rules | `masster_smarts_enveda_gate_2026_09_29.md` |
| [SMARTS pathway gates G0, G1, G2, G2b: every arm fails its gate](#smarts-pathway-gates-g0-g1-g2-g2b-every-arm-fails-its-gate) | 2026-09-30 | G0 +0.00125 < 0.003, G1 STOP, Track B 0 promoted: every arm fails | `masster_smarts_pathway_gates_2026_09_30.md` |

## MASSter SMARTS set and literature review

> **Status:** FINDING · **Updated:** 2026-09-13
> Registry and tests checked; targeted primary-literature review complete. Coverage gains and new atom mappings remain unvalidated.

### Scope and conclusion

This review follows the proposal coverage review (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_proposal_review_2026_09_13.md`)
and its roughly 6% uncovered filtered target intensity. Prioritize candidates applicable to
the current protonated, positive-ion screen. This is a targeted literature check, not an
exhaustive survey or evidence that rearrangements explain the entire deficit.

The [canonical SMARTS catalog](../../reference/masster_smarts_rules.md) retains the complete
rule inventory and admission gates. The main findings are an unaudited legacy eligibility
policy, a concrete pyrimidine gap in the carbamate family, and several positive-ion
oxygen-transfer candidates already described in the literature backlog.

### What the executable set contains

Inspection of [masster_templates.py](../../../src/fragnnet/frag/masster_templates.py)
and its shared [definitions](../../../src/fragnnet/frag/smarts_prepass.py) confirms:

| Set | Count | Eligibility behavior |
| --- | ---: | --- |
| Compound-fragment rules | 12 | Eleven legacy rules plus pyridyl-carbamate transfer. |
| Neutral-loss rules | 14 | Eight legacy rules plus six newer literature templates. |
| Legacy rules | 19 | No enforced precursor-type or fragmentation-mode restrictions. |
| Newer rules | 7 | Explicit precursor restrictions; anisoyl-fluoride additionally requires EI. |

Evaluating the registry eligibility maps gives 23 potentially eligible rules for `[M+H]+`
with HCD, 21 for `[M-H]-` with CID, and 22 for `[M]+` with EI. These counts precede
substructure matching and are not counts of rules experimentally validated under those
conditions. For positive HCD, the 23 comprise 19 unrestricted legacy rules plus pyridyl
carbamate, methyl benzoate, and the two short-chain haloalkyl transfer rules.

The implementation creates precursor-atom partitions and alternative assembly mechanisms;
it does not execute a fully specified product-bond graph followed by arbitrary sequential
fragmentation. Final mass coverage still depends on the ion/H channels. Pure topology
changes that preserve already available atom sets and masses need not expand support.

### Existing-set issues to address before expansion

**Legacy eligibility needs a rule-level audit.** Source comments for sulfonate and diaryl
ketone losses describe negative-ion examples, but the templates are accepted in other
modes too. The test `test_existing_rules_remain_unrestricted_pending_separate_mode_audit`
explicitly preserves this policy. This is an evidence/eligibility gap, not proof that every
cross-mode match is wrong.

In particular, do not automatically restrict sulfonamide SO2 loss to negative mode:
[Sun et al., 2008](https://pubmed.ncbi.nlm.nih.gov/18059004/) report positive-ESI CID
SO2 extrusion in selected arylsulfonamides, with substituent and nitrogen-nucleophilicity
dependence. Eligibility needs scaffold-specific evidence rather than a blanket polarity rule.

**One rule can duplicate ordinary support.** `aryl_sulfonate_so3` retains an ordinary
connected remainder. As the catalog already notes, any benefit may come from forcing its
mask or scoring its identity, rather than adding a new assembly. Measure these separately.

**A matched SMARTS is not necessarily an emitted correct mass.** Check final ion formulas
and allowed hydrogen shifts, then measure added target coverage. Partition and compilation
tests alone cannot establish this scientific outcome.

### Literature priorities for the positive-ion screen

#### 1. Pyrimidinyl-carbamate methyl transfer: a concrete registry gap

The existing pyridyl rule restricts its ring to pyridine patterns. Its original
[positive-ESI CID study](https://www.sciencedirect.com/science/article/abs/pii/S004040201401518X)
supports methyl, ethyl, and n-butyl transfer on pyridyl carbamates. It does not by itself
justify changing the rule to all heteroaromatic carbamates.

A separate experimental [carbamate pesticide study, 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6222809/)
reports pirimicarb's unexpected nominal 57 Da loss and proposes methyl transfer involving
its pyrimidine-containing residue (section 2.3/Figure 6). This supplies a specific lead for
a separately guarded pyrimidine rule, not a generic carbamate transformation.

Direct current-code probes:

- Pyridyl control `CN(C)C(=O)Oc1ccncc1` matches `pyridyl_carbamate_alkyl_transfer`.
- Pyrimidyl analogue `CN(C)C(=O)Oc1cncnc1` matches no registry rules.
- Pirimicarb `CC1=NC(=NC(=C1C)OC(=O)N(C)C)N(C)C` matches no registry rules.
  The equivalent aromatic structure is documented by [ChEBI](https://www.ebi.ac.uk/chebi/CHEBI%3A8248).

**Next:** inspect the product scheme, specify the transferred methyl and retained atom map,
and check whether the observed mass is already covered through another ordinary fragment.
No SMARTS match does not prove zero ordinary spectral coverage. Start with the reported
pirimicarb scaffold and close negative controls before broadening pyrimidine substitution.

#### 2. Phenylalanine-family oxygen transfer

[Zheng et al., 2023](https://www.frontiersin.org/journals/chemistry/articles/10.3389/fchem.2022.1094329/full)
study sequential NH3 and CH2CO elimination leading to oxygen-containing benzyl products.
The paper uses multistage MS, accurate mass, isotope experiments, substituent comparisons,
and calculations. It discusses competing product structures with the same formula.

This is a plausible new heavy-atom assembly for MASSter. Preserve the studied scaffold,
account for the sequential pathway, and verify that the endpoint appears in the target
HCD spectra; an MS3 result alone is not a guarantee of HCD abundance. The publisher notes
a minor correction that does not affect scientific content.

#### 3. Aryl-sulfonamide oxygen transfer

[Guo et al., 2019](https://www.sciencedirect.com/science/article/pii/S1387380618301544)
report protonated sulfonamide fragmentation giving oxygen-containing aryl ions and study
ring-substituent effects. The accessed publisher abstract and experimental snippets support
this as a distinct lead from the implemented SO2-elimination partition. Full product-map
and substituent-guard extraction remains necessary.

Keep this positive-ion family separate from the catalog's negative-ion N-acyl-sulfonamide
carbonyl-oxygen transfer family. Shared words such as “oxygen transfer” are insufficient
to combine their SMARTS or ionization restrictions.

#### 4. Prazole-sulfone SO2 extraction

The positive-ESI [prazole study, 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10420917/)
reports rearranged products from sulfones, while corresponding sulfoxide products were
rare and pyridine N-oxides did not follow that pathway. This gives useful negative controls
for a narrow scaffold rule. Do not translate it into unrestricted sulfone or sulfoxide loss.

Implementation priority among these families depends on how often their scaffolds occur
among the actual missed filtered peaks. That frequency was not measured here.

### Existing rules worth measuring first

Methyl-benzoate CO2/methyl transfer and pyridyl-carbamate alkyl transfer already have
positive-ion literature support; the former is supported by
[Franski et al., 2018](https://pubmed.ncbi.nlm.nih.gov/29441643/). Measure their incremental
coverage before duplicating them with new generic neutral-loss rules. Short-chain halogen
transfer is also already present, with narrow chain-length and precursor restrictions.

Negative-ion Smiles families and the EI-only anisoyl rule remain relevant to their own
datasets, but cannot supply direct evidence for the current protonated filtered screen.
The broader catalog's remaining citations were not all reverified in this targeted pass.

### Other rule collections to compare

A targeted external-set check identifies two priorities beyond the current registry:

| Collection | Evidence and access | Relevance to MASSter |
| --- | --- | --- |
| ChemFrag | [2025 paper](https://doi.org/10.1002/jms.5129): 44 cleavage and 16 rearrangement rules; supporting Tables S1/S2 describe rules. | Inspect heavy-atom migrations, including the steroid methyl-shift example. Many rules act on charged intermediate graphs; conversion to precursor-atom partitions is not automatic. Complete machine-readable rule availability was not verified. |
| CFM-ID MSRB | [CFM-ID 4.0 paper and supplement](https://doi.org/10.1021/acs.analchem.1c01465), [author source repository](https://bitbucket.org/wishartlab/msrb-fragmenter). | Class-specific flavonoid/glycoside rules are a useful comparison. The supplement lists 47 flavonol, 20 flavone, 15 flavanone, and 63 flavonoid-7-O-glycoside rules, among other classes. These are not counts of novel heavy-atom assemblies; ordinary cleavage and ion/H behavior may overlap MASSter. |
| MS-FINDER HR rules | [Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7063832/): nine hydrogen-redistribution rules for CNOPS and irregular-H handling. | Compare against ion/H support, especially P/S behavior; this is not a replacement heavy-atom rearrangement registry. |
| LipidCreator | [Paper](https://www.nature.com/articles/s41467-020-15960-z), [source](https://github.com/lifs-tools/lipidcreator): fragment calculations for over 60 lipid classes. | Prioritize only if lipid scaffolds dominate residuals. Formula/building-block rules require explicit atom-mapping work to become MASSter templates. |
| PPGB-MS2 | [Author repository](https://github.com/zhengfj1994/PPGB_MS2); historical [local audit](../../reference/smarts.md). | Already considered by the project. The local audit reports substantial ordinary-cut/H overlap; broad crf12/crf13 patterns are deliberately excluded from the current registry. Current upstream equivalence was not exhaustively re-audited. |

The ChemFrag article provides an explicit positive-ESI steroid methyl migration from a
dehydrated intermediate. Migration alone preserves mass; only subsequent cleavage can
demonstrate new support. The needed capability may therefore be rearrangement followed
by fragmentation, rather than adding a single precursor-level loss rule.

For the current positive protonated screen, prioritize ChemFrag rearrangements and CFM-ID
flavonoid rules by residual scaffold frequency. CFM-ID's acylcarnitine/acylcholine tables
specify `[M]+`, so they require separate ionization consideration. No complete external
rule-set import, pairwise coverage comparison, or license audit was performed.

### Recommended coverage check

1. On fixed filtered validation IDs, record ordinary candidates and missed target intensity.
2. Add only existing eligible SMARTS candidates and compute union coverage plus incremental
   intensity per rule/scaffold. Do not sum overlapping per-rule gains as if disjoint.
3. Repeat under the production budget to expose displaced ordinary candidates.
4. Check final ion formulas/H shifts and rank residual scaffold families by missed intensity.
5. Pilot the smallest supported new mapped rule. Require positive/negative examples,
   elemental conservation, added support beyond ordinary candidates, and acceptable
   candidate load under the canonical catalog's gates.

A coverage-only diagnostic can establish headroom without training a new intensity model.
It does not establish usable intensities, improved precision, or a validated chemical
assignment for every newly mass-matched peak.

### Validation and limitations

- Paper links are recorded beside all seven newer rule definitions in
  `masster_templates.py`, along with explicitly unimplemented candidate references.
  The canonical catalog now also links the pirimicarb expansion lead. These are citation
  additions only; rule patterns, mappings, indices, and eligibility are unchanged.
- Configured FRAGNNET-GPU interpreter; `pytest tests/test_masster_smarts.py -q`:
  **75 passed** on CPU using the existing local extension.
- Registry eligibility counts and the three substructure probes above were executed.
- Primary literature was checked through accessible full text, abstracts, and publisher
  snippets. New product atom mappings were not validated, and no claim of exhaustive
  literature coverage is made.
- No rule, model, config, data, or environment was changed; no training, real-checkpoint
  coverage evaluation, GPU test, or full-suite run occurred.
- Existing user documentation edits were preserved. Next check is per-rule incremental
  coverage on the filtered validation subset, with pirimicarb and oxygen-transfer families
  retained as targeted leads pending dataset occurrence and product-map checks.

## MASSter SMARTS templates: CUDA enablement, audit and proposal reach

> **Status:** FINDING · **Updated:** 2026-09-28
> SMARTS templates now run on the CUDA candidate backend that every current MASSter config
> uses. This records the port, an audit of the SMARTS path, how often the trained `c0`
> proposal reaches the template cuts on its own, and what coverage they add. The shared-registry
> matcher defects are fixed, MASSter dropped its ring rules and duplicate templates, and AIMe's
> worked examples are checked. Nothing trained beyond one-epoch local smokes. Arm `t1` is
> prepared, not submitted. Uncommitted.

### Answer

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

### 1. What changed

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

### 2. Validation

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

### 3. How often the trained proposal reaches template cuts

Probe: `scripts/debug/smarts/debug_masster_smarts_proposal_reach.py` on the `c0` checkpoint
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

#### 3.1 Coverage the templates add beyond the pool

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

### 4. Audit of the SMARTS path

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

### 5. Decisions (2026-09-27)

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

### 6. AIMe's worked examples and the other rule lists

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

### 7. New rules from the documented lists

At the user's direction, rules were added for the cases plain cuts cannot reach. The
candidates came from:
- the ESI-CID list in [`masster_smarts_rules.md`](../../reference/masster_smarts_rules.md);
- the 09-23 audit's sulfonamide variants;
- AIMe's hydroxamic example.

The ChemFrag triage's only lead, the Wagner–Meerwein migration, still lacks a post-migration
cut and was not encoded. Probe: `scripts/debug/smarts/debug_masster_smarts_rule_gate.py`, whose output is in
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

### 7b. Ring-cut review of 2026-09-21 (from local memory, verified 2026-10-06)

Source: agent-memory review of 2026-09-21, written before the CUDA port above. Its headline, that
templates are a hard CUDA incompatibility (`ValueError` when combined with
`candidate_backend: cuda`), is **superseded** by section 1. The measurements below still stand.

- **No ring-cut code exists in MASSter.** Ring cleavage is emergent: `top_k_cut_masks` ranks arbitrary
  bond masks by independent-Bernoulli probability, then union-find (`build_component_arrays_raw`) or the
  CUDA `connected_components` kernel decides which masks split the graph. A single ring-bond cut is one
  component and is discarded; two or more bonds of one ring are a real cleavage. `max_ring_cut_steps`,
  `ring_open_mask`, `ring_bond_groups` and `GetRingInfo` belong to the FraGNNet DAG BFS only.
- **Ring-cut waste is not a bottleneck.** At E=4096 / U=256 on real NIST20 molecules, non-productive
  (single-component) masks are 3.3-3.4% of scanned masks; at the default E=256 it is 6-18% (up to 34%
  for high-ring molecules under a flat prior, and only 73-82% of molecules reach U=256). NIST20
  ring-bond fraction averages 52.7% and 65% of molecules are at least 50% ring bonds. Eliminating the
  waste is worth about 0.2% end to end (3% of topology x 60% of candidate x 5-10% of batch), against the
  plan's 5% gate.
- **Cost of SMARTS on the CPU path** (real molecules, E=4096 / U=256, batch 128): +19.8% candidate wall,
  `smarts_assembly_ms` 27.8 ms (15.5% of candidate total) for 7 matched molecules, spent rebuilding
  `bond_set_to_mask_index` over all 4096 masks in Python. Reach: 3.8% of NIST20 molecules match any
  ring-cleavage rule (113 templates / 3000 molecules), 5.5-9.0% match any template.
- **Benchmark trap.** `scripts/debug/bench/benchmark_masster_candidates.py` builds ring-free linear
  chains and gives every molecule one synthetic template, so its `smarts_assembly_ms` 1129 ms and 3.5x
  slowdown is a 100%-match worst case; never quote it as the SMARTS cost.
- **Accuracy limit no kernel fixes.** Independent Bernoulli underweights ring cleavage because a
  cleavage needs two correlated cuts, P = p1 x p2, below both singleton ancestors that produce nothing.
  `productive_mask_fraction` lumps redundancy with ring openings, so it is an upper bound on ring waste.
- **Missing-control note.** The old header claim that SMARTS is incompatible with proposal-only warmup
  is not enforced in code: the warmup gate checks only `proposal_selection_only` and
  `proposal_loss_weight > 0`. The SMARTS-off, warmup-null sibling of the SMARTS arm was never built.

### 8. Method and limits

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

## MASSter SMARTS rules and outlier spectra

> **Status:** FINDING · **Updated:** 2026-09-29
> The registered rules matter only on a small set of spectra, where much of the target
> intensity sits at m/z that no ordinary fragment reaches. On `i1`'s `[M+H]+` val set those are
> 37 spectra of 14 molecules, with mean cosine 0.25-0.56. They are moderately bad spectra, not
> the floor: the rules cannot raise `i1`'s lower tail (p5 moves at most 0.0001). The whole-val
> cosine ceiling moves by 0.0001. `[M-H]-` is untested. Model-free; no model trained.

### Question and answer

The user's goal for the SMARTS rules is to help outlier spectra, not the average. The question is
therefore not whether the rules help NIST20 on average, but whether they reach spectra the
model gets badly wrong, and how many.

- **Averaged over the dataset, the rules are negligible.** Summed over all NIST20
  `[M+H]+`/`[M-H]-` spectra, the 14 registered rules' rule-only products explain 0.033% of
  `[M+H]+` and 0.160% of `[M-H]-` target intensity beyond random (the 2026-09-27 gate rows).
- **On outliers, they reach real, unpredictable intensity.** Across NIST20, counting all splits,
  a rule-only product explains:

  | Share of spectrum intensity | `[M+H]+` spectra (molecules) | `[M-H]-` spectra (molecules) | Random null, `[M+H]+` / `[M-H]-` |
  |---|---:|---:|---:|
  | ≥ 10% | 361 (89) | 606 (148) | 18 / 8 |
  | ≥ 25% | 123 (36) | 327 (106) | 4 / 1 |
  | ≥ 50% | 23 (13) | 162 (78) | 2 / 0 |

- **These are `i1`'s outliers, and `i1` cannot fix them.** On `i1`'s `[M+H]+` val set:
  - 37 spectra (14 molecules) have ≥10% of their intensity at rule-only m/z;
  - `i1` puts no predicted intensity there, since it has no candidate at those formulas;
  - their cosine falls as the share rises (table below), and 11 of the 37 sit in the bottom 10%
    of val cosine (below 0.334).
- **Rules involved:** on those 37 spectra, primary sulfonamide O transfer (20), aryl sulfonate
  SO2 (9) and secondary aryl sulfonamide SO2 (8). On NIST20 `[M-H]-`, the same sulfonyl
  family dominates the tail (sulfonamide SO2, sulfonate SO2, primary sulfonamide SO2).

`i1` val `[M+H]+`, template-bearing spectra by rule-only share of linear intensity:

| Rule-only share | Spectra | Molecules | `i1` mean cosine | Cosine cap without the rule peaks |
|---|---:|---:|---:|---:|
| 0 | 1,722 | 146 | 0.725 | 1.000 |
| 0-1% | 102 | 37 | 0.665 | 1.000 |
| 1-5% | 133 | 32 | 0.687 | 0.999 |
| 5-10% | 39 | 17 | 0.750 | 0.989 |
| 10-25% | 31 | 10 | 0.555 | 0.950 |
| 25-50% | 5 | 3 | 0.396 | 0.763 |
| > 50% | 1 | 1 | 0.247 | 0.441 |

The whole val set has 52,020 spectra with mean cosine 0.762. The cosine cap is
`sqrt(1 − s²)`, where `s` is the rule-only share of the spectrum's L2 norm on linear intensity.
It is the best cosine a model can reach while it has no candidate at those m/z. Lifting every
cap to 1 raises the whole-val mean ceiling by 0.00008.

### The lower tail: the rules do not reach it on `[M+H]+`

The user's case for the rules is that they raise the lower bound, the worst spectra. This was
tested on `i1`'s val set with an optimistic upper bound: each spectrum's cosine gains
`1 − cap`, the part it cannot reach without rule candidates.

- **The worst spectra carry no rule-only intensity.** None of the bottom 500 val spectra
  (cosine ≤ ~0.02) has ≥10% rule-only intensity, and none of the worst 100 molecules does
  either.
- **The low quantiles barely move, even under the optimistic bound:**

  | Val cosine quantile | `i1` now | With rules, upper bound |
  |---|---:|---:|
  | p1 | 0.0223 | 0.0223 |
  | p5 | 0.1786 | 0.1787 |
  | p10 | 0.3344 | 0.3346 |
  | Molecule-mean p1 | 0.2085 | 0.2085 |

- **Rule-only peaks are a minority of their own spectra's error.** They account for 18.8% of
  the cosine gap (`1 − cos`) on the 37 spectra with ≥10% rule-only intensity. `i1` gets the
  rest of those spectra wrong for other reasons.
- **So on `[M+H]+` the rules target moderately bad spectra (cosine 0.25-0.56), not the floor.**
  Whatever drives the near-zero-cosine spectra, these rules do not touch it.
- **`[M-H]-` is untested.** Its rule-only tail is larger (162 spectra at ≥50% rule-only
  intensity, against 23 in `[M+H]+`), and no `[M-H]-` model exists to rank its spectra.
  Whether the rules reach the `[M-H]-` floor needs that model's val cosines.

Log: `output/smarts_outliers_2026_09_29/lower_tail_log.txt`.

### What this means for judging SMARTS arms

- **A whole-split metric cannot show the effect.** Even a perfect use of the rules moves the
  `[M+H]+` val ceiling by about 0.0001, far below the ~0.004 seed spread.
- **The effect must be read on the template-bearing subset and its tail.** For example, cosine
  on spectra with rule-only share ≥10%, compared with a templates-off control on the same
  spectra.
- **The tail is about four times as frequent in `[M-H]-`:** 0.39% of its spectra reach ≥10%
  (606 of 156,532), against 0.10% in `[M+H]+` (361 of 367,379). `[M-H]-` has no trained model
  yet.
- **G0 in the SMARTS pathway plan (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_smarts_pathway_plan.md`)** gates
  `t1i` on the whole-split coverage ceiling. That gate measures the average, not the outliers
  this goal targets.

### Method and limits

- **Rule-only m/z:** the per-class gate's unique channels. These are the channels of a rule's
  disconnected product that no connected atom set of the same heavy-atom composition reaches
  within 4 H, matched at 10 ppm with a 200 m/z floor. Rules are the 14 registered ones, each in
  its admitted polarity; per spectrum, the union over rules.
- **Dataset-level and tail counts** come from the 2026-09-27 gate rows
  (`output/smarts_reach_2026_09_27/rule_gate*.pkl`). There each rule's rows follow the
  molecule's spectrum order, so a spectrum's value is the maximum over its rules, a lower bound
  on the union.
- **`i1` join:** from `output/ev_ladder_census_2026_09_28/i1_val.pkl` (per-spectrum `cos_ref`
  and peaks). The cap uses linear intensity, not the metric's own transform, so it is
  approximate.
- **Coverage, not model gain:** a templates-on model still has to rank those candidates. The
  arm is the test.
- **Artifacts:** `output/smarts_outliers_2026_09_29/`, holding `i1_rule_outliers.py`, its log
  and `i1_val_rule_outliers.pkl`.

## MASSter SMARTS rules on Enveda-180: per-class gate

> **Status:** FINDING · **Updated:** 2026-09-29
> Model-free coverage gate, the NIST20 measure, run on every SMARTS rule in the code. Five of
> 41 rules add intensity beyond random on Enveda-180, all of them S=O rules; only one reaches
> 1%. None of the eight rules disabled on 2026-09-29 finds support. No model was trained.

### Answer

The question was whether the SMARTS rules, including the eight disabled on 2026-09-29 to
re-check here, are useful on Enveda-180 (Bruker timsTOF, `QTOF`/`CID`, Enamine molecules).

- **5 of 41 rules pass** (95% CI above 0 in an admitted polarity):
  - `primary_aryl_sulfonamide_so2`;
  - `aryl_sulfonamide_so2`;
  - `sulfoxide_so_loss`;
  - `primary_aryl_sulfonamide_o_transfer`;
  - `tertiary_aryl_sulfonamide_so2`.
- **Only one reaches 1% of intensity:** primary sulfonamide SO2 in `[M-H]-`, +1.48%. The
  others are +0.05% to +0.82%.
- **The rules are 2-10x weaker than on NIST20.** For example, `aryl_sulfonamide_so2` gives
  +0.82% in `[M-H]-` here against +3.75% on NIST20, and primary sulfonamide O transfer gives
  +0.23% in `[M+H]+` against +2.61%.
- **None of the eight disabled rules is useful here:**
  - five match no Enveda-180 molecule;
  - methyl benzoate (1,269 molecules) and diarylvinyl (189) score 0.00;
  - diaryl ketone cannot match: the library has no aryl ketone (0 of 183,476 molecules).
- **Sulfoxide is the one change of verdict.** On NIST20 it scored −0.05 / 0.00; here it gives
  +0.56% in `[M-H]-` (CI 0.23 to 1.01, 208 molecules). It is one of the three failing rules the
  catalog keeps pending an ablation.
- **25 rules match no molecule at all**, including eight registered rules: aryl sulfonate,
  nitramine, C2 haloalkyl, N-acyl sulfonamide, diacylhydrazine, phenylalanine, hydroxamic acid
  and PFAS sulfonate.

### Method

Probe: [`debug_masster_smarts_rule_gate_enveda.py`](../../../scripts/debug/smarts/debug_masster_smarts_rule_gate_enveda.py).
It reuses the NIST20 gate's helpers from `debug_masster_smarts_rule_gate.py`: the product sets,
the connected-twin search and MASSter's channel function. So the measure is the
[per-class gate](../../reference/masster_smarts_rules.md#rules-added-by-the-per-class-gate-2026-09-27):
- the target intensity a rule's disconnected product explains through MASSter's channels
  (valence bound, ±4 H, no CMF);
- only on peaks that no connected atom set of the same heavy-atom composition reaches within
  4 H;
- against random disconnected atom sets of the same size;
- averaged per molecule, with molecule-bootstrap 95% CIs.

What differs from the NIST20 run:
- **Rules:** 41, every SMARTS rule in the code. That is:
  - the 14 registered MASSter rules;
  - the 8 disabled on 2026-09-29, from copies of their commented-out definitions;
  - the 12 dropped on 2026-09-27;
  - the 5 shared rules MASSter never used (`crf12_0`, `crf12_1`, `crf13_2`,
    `linear_tricyclic_bis_c4_loss`, `morphinan_ring_cleavage`);
  - the probe's 2 other candidates.
- **Polarity:** every rule is scored in both `[M+H]+` and `[M-H]-`. Verdicts use the polarities
  the rule is (or was) admitted for.
- **Sample:** the census covers all 177,341 molecules with an `[M+H]+` or `[M-H]-` spectrum.
  The gate scores up to 1,000 matched molecules per rule, drawn with a fixed seed.
- **Tolerance:** 10 ppm with a 200 m/z floor, the Enveda configs' QTOF loss tolerance. It is
  also the NIST20 run's OB/OB_F tolerance.

Verdict labels: **pass** means the CI's lower bound is above 0; **clear** means a pass at 1% or
more. The NIST20 catalog has no written threshold.

### Results

Rules with at least one matched Enveda-180 molecule. Values are % of intensity beyond random
(95% CI); `*` marks a polarity the rule is not admitted for.

| Rule | Group | Molecules matched | Scored `[M+H]+` / `[M-H]-` | `[M+H]+` | `[M-H]-` | Verdict |
|---|---|---:|---:|---|---|---|
| `primary_aryl_sulfonamide_so2` | registered, `[M-H]-` | 822 | 704 / 659 | +0.02 (−0.00 to 0.04)* | +1.48 (1.02 to 2.02) | pass, clear |
| `aryl_sulfonamide_so2` | registered, unguarded | 7,245 | 808 / 821 | +0.16 (0.10 to 0.23) | +0.82 (0.56 to 1.13) | pass |
| `sulfoxide_so_loss` | registered, unguarded | 729 (565 with product) | 517 / 208 | +0.05 (0.02 to 0.09) | +0.56 (0.23 to 1.01) | pass |
| `primary_aryl_sulfonamide_o_transfer` | registered, `[M+H]+` | 822 | 704 / 659 | +0.23 (0.16 to 0.31) | +0.06 (0.03 to 0.10)* | pass |
| `tertiary_aryl_sulfonamide_so2` | registered, unguarded | 7,135 | 931 / 324 | +0.05 (0.02 to 0.10) | +0.15 (0.03 to 0.34) | pass |
| `n_phenylbenzamide_co_smiles` | registered, `[M-H]-` | 8,295 | 944 / 690 | 0.00* | 0.00 | fails |
| `methyl_benzoate_co2_transfer` | disabled, `[M+H]+` | 1,269 | 903 / 449 | −0.01 (−0.02 to 0.00) | 0.00* | fails |
| `diarylvinyl_c2h2_loss` | disabled, unguarded | 189 | 179 / 124 | 0.00 | 0.00 | fails |
| `halogenated_quinoline_ring_cleavage` | dropped 09-27 | 883 | 855 / 317 | −0.00 (−0.01 to 0.00) | 0.00 | fails |
| `benzopyranone_ring_cleavage` | dropped 09-27 | 124 | 112 / 55 | −0.03 (−0.07 to 0.00) | +0.01 (0.00 to 0.02) | fails |
| `benzopyranone_bis_ch_loss` | dropped 09-27 | 124 | 113 / 55 | 0.00 | 0.00 | fails |
| `crf12_0` | shared, never MASSter | 176,076 (155,312 with product) | 943 / 390 | +0.01 (−0.00 to 0.02) | −0.01 (−0.04 to 0.00) | fails |
| `crf12_1` | shared, never MASSter | 2,352 (600) | 570 / 264 | −0.01 (−0.05 to 0.01) | 0.00 (−0.00 to 0.00) | fails |
| `crf13_2` | shared, never MASSter | 5,828 (5,446) | 949 / 302 | 0.00 (−0.00 to 0.01) | +0.01 (0.00 to 0.03) | fails |
| `linear_tricyclic_bis_c4_loss` | shared, never MASSter | 34 | 32 / 6 | 0.00 | 0.00 | fails |
| `n_methylanilinoethanol_o_smiles` | candidate, `[M-H]-` | 27 | 26 / 5 | 0.00* | −0.03 (−0.08 to 0.00) | fails |

**No matched molecule (25 rules):**
- **Registered:** `aryl_sulfonate_so2`, `n_nitramine_n2o`,
  `short_haloalkyl_ammonium_c2_transfer`, `n_acyl_aryl_sulfonamide_carbonyl_o_transfer`,
  `diacylhydrazine_carboxylate_o_transfer`, `phenylalanine_carboxyl_o_transfer`,
  `hydroxamic_acid_nh_excision`, `pfas_sulfonate_f_transfer`.
- **Disabled 2026-09-29:** `diaryl_ketone_co_loss`, `aminobenzothiazole_hcn_loss`,
  `pyridyl_carbamate_alkyl_transfer`, `anisoyl_fluoride_co_f_transfer`,
  `short_haloalkyl_ammonium_c3_transfer`, `n_fluoro_sulfonamide_f_transfer`.
- **Dropped 2026-09-27:** both anthraquinone rules, xanthone, phenoxazine, furanocoumarin, the
  morphinan arm and both desoxymorphinan rules, and `aryl_sulfonate_so3`.
- **Other:** `morphinan_ring_cleavage` and `pfas_sulfinate_f_transfer`.

**The zeros reflect the library, not the matcher.** Direct RDKit counts:
- no aryl ketone (`c-C(=O)-c`) among all 183,476 molecules;
- no ketone and no N-OH group in the first 20,000;
- in that same sample, 86% carry an amide, 11% a carboxylic acid and 7% an aryl sulfonamide.

**Reach:** products of the five passing rules exist for 15,652 molecules. That covers 7.8% of
the 590,683 `[M+H]+` spectra and 13.8% of the 185,030 `[M-H]-` spectra.

### Limits

- **Coverage, not model gain.** The gate bounds what a rule could add at the formula level.
  Pool reach (NIST20 [finding §3.1](#31-coverage-the-templates-add-beyond-the-pool))
  needs a trained Enveda proposal and was not measured.
- **Adducts:** only `[M+H]+` and `[M-H]-` are scored, 67% of Enveda-180 spectra. Dimers
  (28.7%), `[M+Na]+`, formate and the other adducts are not.
- **Peaks:** the proc tables come from the release's filtered export, which drops peaks below 1%
  of the base peak ([dataset finding §2](../enveda_180_dataset_2026_09_25.md#2-the--filtered-export-is-a-1-base-peak-floor)).
- **Sample:** rules with more than 1,000 matched molecules are sampled. At most 13% of a
  rule's scored spectra have a product that hit the twin search's cap
  (`benzopyranone_bis_ch_loss`); a capped product counts as covering nothing.
- **Channels:** as in the NIST20 run, a product's valence bound takes no slots back for its
  newly formed bonds, so its channels are an upper bound.

### Provenance

- **Command:**
  `python scripts/debug/smarts/debug_masster_smarts_rule_gate_enveda.py --workers 10 --max_molecules 1000`.
  It took 265 s on the workstation.
- **Outputs:** in `output/smarts_gate_enveda_2026_09_29/`: `census.pkl`, `rule_gate.pkl`,
  `summary.csv`, `run.log`.
- **Revision:** `git_rev.txt` records the HEAD revision and the dirty diff stat. The working
  tree held the 2026-09-29 rule disabling and another session's uncommitted SMARTS pathway work.
- **Data:** `data/proc/enveda_180/{mol_df,spec_df}.pkl`
  ([dataset finding](../enveda_180_dataset_2026_09_25.md)).

## SMARTS pathway gates G0, G1, G2, G2b: every arm fails its gate

> **Status:** FINDING · **Updated:** 2026-09-30
> Phase 1 of the SMARTS pathway plan (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_smarts_pathway_plan.md`), run
> 2026-09-29/30 on the local workstation. The decision rules were fixed before any result, and
> two adversarial verifiers per gate recomputed the headline numbers from the artifacts. They
> found no blocking issue.
>
> - **Scope:** NIST20 only. MONA is excluded (user decision, 2026-09-30).
> - **Registry:** 14 active rules. The vocabulary stays at 22.
> - **Test split:** never inspected.

### Decision

| Arm | Gate | Pre-registered rule | Result | Verdict |
|---|---|---|---|---|
| `t1i` on `[M-H]-` | G0 | whole-split coverage-ceiling rise ≥ +0.003 | +0.00125 [0.00091, 0.00163] | **FAIL** |
| `p1` (Track A) | G1 | headroom ≥ +0.003 and `rules` over `context` CI lower > 0 | headroom −0.000012 [−0.000040, +0.000013]; gain +0.000026 [−0.000020, +0.000084] | **STOP** (CI half-widths ≪ 0.003, so not underpowered) |
| `p1c` (Track B) | G2/G2b | promoted families + model operators reach a ceiling ≥ +0.003 | 0 families promoted, 0 operators | **FAIL** |

`t1i` on `[M+H]+` ran only as `p1`'s control, gated on "G1 continues", so it is not warranted
either. No Phase 2 arm of the plan has a gate that passes.

### Scope and units

- **Coverage ceiling.** Per spectrum, w is the eval-convention target: L1-normalised, linear,
  unbinned. With N = Σw², A0 is the w² the baseline covers and dA the w² newly covered. The rise
  is √((A0+dA)/N) − √(A0/N), averaged over every spectrum of the stratum (spectra with nothing
  new contribute 0). CIs are a molecule-clustered bootstrap.
- **Candidate-space, not W&B.** The rise is an upper bound on the `cos_sim_0.01` gain, not the
  W&B metric.
- **Production channels.** Template sides use cut count `len(broken_bond_indices)` and
  `formed_bond_count`, the valence bound, ±4 H and no CMF. This is the channel function the
  Phase 0 code uses in the model.
- **Unique coverage.** A peak counts only if no connected atom set of the same heavy-atom
  composition reaches it within ±4 H (`reachable_h`, cap 200,000). Capped searches earn no
  credit.
- **`[M-H]-` molecule assignment.** From `data/split/nist20_ma_mi_inchikey`, via 2D InChIKey,
  restricted to model-free use. That split and `i1`'s `nist20_inchikey` are independent draws.
  Of the `ma_mi`-val `[M-H]-` molecules that also have `[M+H]+` spectra, 1,108 are in `i1`'s
  train. So nothing trained on `i1`'s split may be scored on it (open decision D1).

### G0: `[M-H]-` templates on

**Headline stratum:** NIST20 OB/OB_F `[M-H]-` spectra, train, val and test pooled.
- 121,053 spectra from 10,833 molecules.
- 6,823 spectra (5.6%) from 568 molecules carry an eligible template.
- Only 1,360 spectra (1.1%) gain anything.

| Stratum | Spectra | Rise [95% CI] | Rise on template spectra (D2) |
|---|---:|---|---:|
| **Headline** | 121,053 | **0.00125 [0.00091, 0.00163]** | 0.0222 |
| train | 72,310 | 0.00125 [0.00079, 0.00178] | 0.0208 |
| val | 24,540 | 0.00141 [0.00073, 0.00216] | 0.0303 |
| test (census only) | 24,203 | 0.00109 [0.00049, 0.00188] | 0.0197 |
| NIST20 QTOF | 2,394 | 0 | 0 |

**Bounds.**
- The baseline is a stand-in pool: all 0-, 1- and 2-cut components. It over-covers, with A0/N
  a median 0.99 on template spectra.
- Without it, the rise is at least dA/(2N) = 0.00080.
- A fully pool- and twin-free upper bound, √(dA/N) with A0 = 0 and no twin exclusion, is
  **0.00269** (verifier, point estimate). The gate fails whatever the real pool is.
- The random disconnected null gives 3×10⁻⁷.

**Sensitivities.** The headline stays within 0.00011:
- no twin exclusion: 0.00136;
- optimistic valence: 0.00125, since the formed-bond correction never binds here;
- capped sides credited: 0.00125.

**Concentration.** The rise is heavy-tailed: 100 spectra hold 58% of it.

**Per rule** (rule alone, headline stratum):

| Rule | Molecules | Rise | L1 pp on its spectra |
|---|---:|---:|---:|
| `aryl_sulfonamide_so2` | 192 | 0.00059 | 3.95 |
| `aryl_sulfonate_so2` | 85 | 0.00031 | 6.08 |
| `primary_aryl_sulfonamide_so2` | 51 | 0.00028 | 6.21 |
| `n_acyl_aryl_sulfonamide_carbonyl_o_transfer` | 7 | 0.00005 | 7.26 |
| `n_nitramine_n2o` | 2 | 0.00002 | 16.29 |
| `tertiary_aryl_sulfonamide_so2`, `n_phenylbenzamide_co_smiles`, `sulfoxide_so_loss` | 53 / 173 / 16 | ≈ 0 | ≈ 0 |
| `diacylhydrazine_carboxylate_o_transfer`, `pfas_sulfonate_f_transfer` | 0 | — | — |

- The two largest `[M-H]-` per-class effects in the plan's §1 (diacylhydrazine +49.6, PFAS)
  come only from MONA spectra. With MONA excluded, they have no evidence.
- Validation: with optimistic valence and no pool, the scorer reproduces the 2026-09-27
  per-class gate's `unique_fraction` exactly on 1,005 spectra.

### G1: rule provenance on `i1` (`[M+H]+`)

- **Run:** `i1` epoch 59 on val (52,020 spectra), with a fresh extract that saves candidate m/z
  and rule columns.
- **Primary marker:** `rule_any`, a candidate peak at a valid production channel of either side
  of an eligible `[M+H]+` template (1e-5 Da).
- **Coverage:** 187 val molecules (4.4%) and 2,598 spectra carry a template. After the save
  floor, 1,230 marked rows in 611 spectra and 83 molecules remain.
- **Target mass:** marked peaks hold 0.000225 of target mass per spectrum, and `i1` already
  predicts 0.000240 on them.

| Family (cross-fitted, candidate-space cos) | MLP over `context` [CI] | Linear over `context` [CI] |
|---|---|---|
| `existing` | +0.000589 [+0.000120, +0.000962] | +0.000191 [−0.000139, +0.000435] |
| `rules` | +0.000026 [−0.000020, +0.000084] | +0.000001 [−0.0000005, +0.0000038] |
| `rules+existing` | +0.000615 [+0.000184, +0.000949] | +0.000192 |
| `rules_permuted` (control) | −0.000006 | −0.000018 [−0.000034, −0.000005] |

- **Ceiling.** The literal swap of every marked peak to its target share gains only +0.000019
  whole-split before control subtraction, about 150× below +0.003.
  - The matched control gains more (+0.000031), so the headroom is slightly negative.
  - The sign partly reflects matching on m/z band and target rank but not on predicted share.
    The decision does not depend on it.
- **Informative marker.** `rule_disc`, twins of a disconnected template side (30 molecules),
  has headroom +3×10⁻⁹ [−2.3×10⁻⁵, +2.1×10⁻⁵].
- **Scope.** G1 tests flags on rule peaks that `i1` already produces as candidates. Rule
  products `i1` cannot produce are G0's and G2's question.

### G2 and G2b: `[M-H]-` operators on fragments

**Setup.** Model-free: the stand-in pool replaces a checkpoint pool. Strict (production-matched)
valence is primary. The baseline is the pool plus the eligible registered template products
plus the generic whole-precursor bridge products.

**G2, rule-restricted continuations.** X is limited to the bridge-shaped registered motifs plus
aryl nitro NO. Train used all 456 motif-bearing train molecules (1,810 spectra); val used 136
molecules (537 spectra).
- **Selection:** the frozen rule selected **0 families**.
  - Four families beat the null on train: nitro ×2 (+1.20, +1.55 pp), sulfonamide NH (+3.86)
    and sulfonate OH (+3.44), strict lower limits > 0.
  - All four were rejected by the no-capped-searches rule: 60–81 capped real sets each.
  - At a cap of 1,000,000 the selection is still 0.
- **Post-hoc, not decision-bearing:** a cap-tolerant Bonferroni check confirms all four on val,
  but their load is 26–41 formulas per bearing spectrum, against a limit of 10.
- **Ceiling (descriptive, every motif family):** val 0.00104 [0.00051, 0.00168], with a
  pool-free upper bound of 0.00332. Train is 0.00162.

**G2b, generic operators.** 512 random train and 512 random val molecules; val has 1,980 spectra.

| Operator | Strict diff vs null, val (pp) [CI] | Load (formulas / spectrum) | Any-cap spectra | Model operator |
|---|---|---:|---:|---|
| bridge (≤3 heavy atoms) | −0.033 [−0.205, +0.158] | 177 | 8.4% | no |
| graft (terminal O/F, ≤4 bonds) | +0.004 [−0.096, +0.108] | 50 | 4.5% | no |

- **Real hits equal random ones.** Real bridge products reach 0.411 pp of unique intensity,
  random disconnected sets 0.443 pp. So the operators' descriptive ceilings are mostly chance.
  - bridge 0.0018, graft 0.0010, union 0.0023 on val;
  - bridge 0.0062, union 0.0049 on train;
  - none is null-adjusted.
- **Graft null.** It is size-matched only and gets about 9% more channels than real grafts, a
  mild leniency. The CI is still ±0.10 pp.
- **Discovery candidates only.** Graft of the O next to an aryl nitro N has load about 6 and
  22 train / 18 val molecules, and it misses on caps. It needs a chemistry read before any rule
  is written.

**Protocol limits found.** Both are pre-registered, and only the user should change them.
- **Load rule:** as measured on the stand-in pool, which inflates it, ≤10 formulas per bearing
  spectrum fails every family and operator by 2.5–20×.
- **No-caps rule:** zero capped searches is effectively unsatisfiable for families with 40–55
  atom molecules.
- **Neither changes Track B.** Even every motif family together stays at a val ceiling of
  0.00104 under the stand-in baseline.

### What this does not show

- **`t1i` gain is untested, as intended.** G0 bounds what `t1i` could add to `[M-H]-` coverage.
  It does not measure a trained model.
- **The D2 read is still open.** The bearing-subset value is 0.022 for G0; whether a gain
  confined to template spectra is worth pursuing is the user's call.
- **G1 is `[M+H]+` only.** No `[M-H]-` checkpoint exists, and the five `[M-H]-`-only rules have
  no `[M+H]+` val molecules.
- **The stand-in pool bounds Track B only loosely.** It is not the real candidate pool, which
  would cover less.

### Artifacts and code

**Artifacts:** [`output/smarts_pathway_gates_2026_09_29/`](../../../output/smarts_pathway_gates_2026_09_29/).
- Reports: `g0_report.md`, `g1_report.md`, `g2_report.md`, `g2b_report.md`.
- Data: per-spectrum rows, fits, ceilings, families json, provenance.
- Slices: `mh_spec.pkl`, `split_map.csv`, `spec_split.csv`.

**Code** (all under `scripts/debug/`, CPU except the G1 extract):
- [`debug_masster_smarts_g0_ceiling.py`](../../../scripts/debug/smarts/debug_masster_smarts_g0_ceiling.py) (new).
- [`debug_masster_postcut_information_gate.py`](../../../scripts/debug/masster_memorization/debug_masster_postcut_information_gate.py):
  rule columns, `--template_bearing_only`, the rule families and a `ceiling` step.
- [`debug_masster_excision_continuations.py`](../../../scripts/debug/smarts/debug_masster_excision_continuations.py):
  polarity plumbing, model-free stand-in pool, motif restriction, graft, confirmation step and
  the ceiling unit.

**Tests:** `tests/test_debug_masster_smarts_g0_ceiling.py`,
`tests/test_debug_masster_postcut_information_gate.py`,
`tests/test_masster_excision_continuations.py` and
`tests/test_debug_masster_pathway_continuations.py` pass (104 tests in total).

**Provenance.** The runs used the uncommitted Phase 0 tree, and `HEAD` moved from `a7aff613` to
`9d55d4a0` during G2b, through another session's commit. Rule names and the gate scripts'
source hashes are identical across all runs.
