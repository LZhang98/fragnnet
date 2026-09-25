# MASSter SMARTS set and literature review

> **Status:** FINDING · **Updated:** 2026-09-13 · Registry and tests checked; targeted primary-literature review complete. Coverage gains and new atom mappings remain unvalidated.

## Scope and conclusion

This review follows the [proposal coverage review](masster_proposal_review_2026_09_13.md)
and its roughly 6% uncovered filtered target intensity. Prioritize candidates applicable to
the current protonated, positive-ion screen. This is a targeted literature check, not an
exhaustive survey or evidence that rearrangements explain the entire deficit.

The [canonical SMARTS catalog](../../reference/masster_smarts_rules.md) retains the complete
rule inventory and admission gates. The main findings are an unaudited legacy eligibility
policy, a concrete pyrimidine gap in the carbamate family, and several positive-ion
oxygen-transfer candidates already described in the literature backlog.

## What the executable set contains

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

## Existing-set issues to address before expansion

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

## Literature priorities for the positive-ion screen

### 1. Pyrimidinyl-carbamate methyl transfer: a concrete registry gap

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

### 2. Phenylalanine-family oxygen transfer

[Zheng et al., 2023](https://www.frontiersin.org/journals/chemistry/articles/10.3389/fchem.2022.1094329/full)
study sequential NH3 and CH2CO elimination leading to oxygen-containing benzyl products.
The paper uses multistage MS, accurate mass, isotope experiments, substituent comparisons,
and calculations. It discusses competing product structures with the same formula.

This is a plausible new heavy-atom assembly for MASSter. Preserve the studied scaffold,
account for the sequential pathway, and verify that the endpoint appears in the target
HCD spectra; an MS3 result alone is not a guarantee of HCD abundance. The publisher notes
a minor correction that does not affect scientific content.

### 3. Aryl-sulfonamide oxygen transfer

[Guo et al., 2019](https://www.sciencedirect.com/science/article/pii/S1387380618301544)
report protonated sulfonamide fragmentation giving oxygen-containing aryl ions and study
ring-substituent effects. The accessed publisher abstract and experimental snippets support
this as a distinct lead from the implemented SO2-elimination partition. Full product-map
and substituent-guard extraction remains necessary.

Keep this positive-ion family separate from the catalog's negative-ion N-acyl-sulfonamide
carbonyl-oxygen transfer family. Shared words such as “oxygen transfer” are insufficient
to combine their SMARTS or ionization restrictions.

### 4. Prazole-sulfone SO2 extraction

The positive-ESI [prazole study, 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10420917/)
reports rearranged products from sulfones, while corresponding sulfoxide products were
rare and pyridine N-oxides did not follow that pathway. This gives useful negative controls
for a narrow scaffold rule. Do not translate it into unrestricted sulfone or sulfoxide loss.

Implementation priority among these families depends on how often their scaffolds occur
among the actual missed filtered peaks. That frequency was not measured here.

## Existing rules worth measuring first

Methyl-benzoate CO2/methyl transfer and pyridyl-carbamate alkyl transfer already have
positive-ion literature support; the former is supported by
[Franski et al., 2018](https://pubmed.ncbi.nlm.nih.gov/29441643/). Measure their incremental
coverage before duplicating them with new generic neutral-loss rules. Short-chain halogen
transfer is also already present, with narrow chain-length and precursor restrictions.

Negative-ion Smiles families and the EI-only anisoyl rule remain relevant to their own
datasets, but cannot supply direct evidence for the current protonated filtered screen.
The broader catalog's remaining citations were not all reverified in this targeted pass.

## Other rule collections to compare

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

## Recommended coverage check

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

## Validation and limitations

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
