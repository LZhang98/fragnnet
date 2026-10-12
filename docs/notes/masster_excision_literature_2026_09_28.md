# Internal-group excision and atom transfer for MASSter

> **Status:** NOTE · **Updated:** 2026-09-28 · Targeted primary-literature search and code
> review completed. Candidate recommendations are untested; no model or dataset changed.

## Scope and conclusion

Question: which experimentally supported gas-phase rearrangements justify extending MASSter
beyond ordinary connected precursor atom sets? This search follows the AIMe excision discussion.
It covers internal neutral losses, Smiles rearrangements, and remote O/F/group transfer under
ESI and collisional activation, with EI and metal-complex evidence explicitly distinguished.
Search terms included internal excision, sulfonamide SO2 loss, hydroxamic NH loss,
Smiles rearrangement, oxygen transfer, isotope labeling, and computational fragmentation.
This is a targeted search, not an exhaustive systematic review. Sources were checked on
2026-09-28; some publisher pages supplied only abstracts or indexed excerpts.

The literature supports restricted rearrangement families. It does not establish the accuracy
or efficiency of an unrestricted delete-and-reconnect operator for MASSter. Strong evidence
comes from isotope labeling, MSn, reference compounds, and substitution controls; a product
formula alone cannot establish its atom mapping or reaction mechanism.

The current implementation represents **precursor atom partitions**, including disconnected
retained assemblies, rather than newly bonded product graphs with sequential fragmentation.
See the [implementation audit](../findings/masster/masster_smarts_review_2026_09_13.md) and
[registry](../../src/fragnnet/models/masster/path_selection/smarts_templates.py). Thus adding candidate support and
learning the actual rearranged connectivity are different implementation scopes.

## Papers and implications

| Study | Observations and evidence inspected | Implication for MASSter |
| --- | --- | --- |
| Wang et al., 2003, [SO2 loss from sulfonamides](https://doi.org/10.1002/rcm.877); [author-hosted abstract](https://scholars.duke.edu/publication/774873) | Negative-mode CID, exact mass, and further fragmentation support SO2 loss with a covalently connected product. The detailed mechanism is proposed. | Direct precedent for internal-group excision; this family is already represented. It does not validate arbitrary sulfur-group deletion. |
| Liang et al., 2021, [N-acyl aromatic sulfonamides](https://doi.org/10.1021/jasms.0c00470); [NIST account](https://www.nist.gov/publications/cid-fragmentation-deprotonated-n-acyl-aromatic-sulfonamides-smiles-type-and-nitrogen) | Negative-mode CID forms phenoxide; 18O labeling identifies the remote amide oxygen, and a thio analogue supports the transfer assignment. DFT supports competing energy-dependent routes. | Strong support for the existing carbonyl-O-transfer rule. The thio analogue is a possible narrow follow-up, subject to corpus support and formula coverage. |
| 2019, [oxygen transfer from protonated sulfonamides](https://doi.org/10.1016/j.ijms.2018.09.033) | ESI-MSn and DFT support phenoxonium formation; ring electronics strongly affect abundance. The example includes NH3 loss before MS3. | Already represented. A broad precursor pattern needs learned suppression or evidence-based restrictions; positive and negative pathways must remain distinct. |
| Li et al., 2015, [deprotonated N-phenylbenzamides](https://doi.org/10.1002/rcm.7172) | Ion-trap MSn, accurate-mass QTOF, and DFT support competing N- and O-initiated routes. CO loss accompanies N attack; para NO2, COCH3, or CF3 on the aniline ring favors a phenolate product through O attack. | The executable registry has the CO branch, but no dedicated O-transfer branch for this scaffold. A specific new candidate to map and screen; the full schemes are needed before encoding atom identities. |
| Liang et al., 2022, [N-to-O Smiles rearrangement](https://doi.org/10.1021/jasms.2c00210); [article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10959088/) | 18O-labeled N-methylanilinoethanol transfers terminal O into phenoxide. The N-H analogue favors a different product; N-ethyl weakens the rearrangement. Morpholinylbenzoic acids undergo a multistep extension. | Strong chemistry, but the existing local gate for the acyclic family was inconclusive. Revisit its capped search before adding a rule. Screen morpholine derivatives separately, preserving their decarboxylation and ring-opening conditions. |
| Liu et al., 2026, [PFAS F migration](https://doi.org/10.1002/rcm.70035) | Negative HR-ESI-MS/MS, AIF, and calculations support CF3SO3− → FSO3− + CF2 and CF3SO2− → FSO2− + CF2. | The sulfonate rule exists. Sulfinate is a separate lead, previously without NIST20 support; this paper does not justify a general fluorine-transfer rule. |
| Yin et al., 2021, [benzyl phenyl sulfones](https://doi.org/10.1002/jms.4691); [abstract](https://pubmed.ncbi.nlm.nih.gov/33295108/) | ESI-MSn and DFT support benzyl transfer. A synthesized benzyl benzenesulfinate reference shows matching gas-phase behavior. | Already in the backlog. Extract the downstream charged product map before implementation: rearranging an intact precursor alone adds no fragment mass. Full schemes were not retrieved in this search. |
| Zhang et al., 2012, [sulfonylurea herbicides](https://doi.org/10.1002/cjoc.201200610); [publisher contents and summary](https://onlinelibrary.wiley.com/toc/16147065/2012/30/10) | Negative ESI shows competing N- and O-initiated Smiles channels for rimsulfuron. | Existing backlog with unresolved product maps. The evidence supports named scaffolds, not every sulfonylurea. Full schemes remain to be inspected. |
| Wang, Nibbering and Bowie, 2010, [phenoxyalkanol anions](https://doi.org/10.1039/C0OB00064G) | 18O evidence and reaction-coordinate calculations show oxygen exchange through Smiles chemistry for selected chain lengths; the four-carbon system favors a different pathway. | A useful negative control: exchanging which O reaches phenoxide can leave the same unlabeled formula and an ordinary connected formula twin. Chemical novelty alone does not establish additional mass coverage. |
| [Zinc-triggered Lossen rearrangement, 2009](https://pubmed.ncbi.nlm.nih.gov/19937618/) | Metal-complex MS, labeling, and calculations support a zinc-assisted hydroxamic-acid rearrangement with water or methylisocyanate elimination. | Does not independently validate AIMe's NH excision of an ordinary protonated hydroxamic acid. Metal state, product, and neutral loss differ. |
| [CFM-ID 4.0, 2021](https://doi.org/10.1021/acs.analchem.1c01465); [article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9064193/) | Combines learned fragmentation with class-specific SMIRKS rules; the expanded rule module contains 313 rules across 11 classes and three adduct types. | Precedent for constrained hybrid modeling, not evidence that excision alone improves accuracy. Its combined improvements do not isolate the effect of rearrangements. |

A recent [2026 sulfonamide preprint](https://doi.org/10.2139/ssrn.7102858) also reports that
N-hydroxylation changes competition between SO2 loss and oxygen transfer. Only its abstract
was inspected; it supports keeping oxidation state explicit, not adding a rule from the
abstract. The NMR evidence establishes the impurity structure, not every proposed gas-phase
intermediate.

## What this changes in our interpretation

**Correction to earlier state:** `hydroxamic_acid_nh_excision` is already in the executable
registry, guarded to `[M+H]+`. The initial rejection in the
[SMARTS audit §6](../findings/masster/masster_smarts_cuda_audit_2026_09_27.md#6-aimes-worked-examples-and-the-other-rule-lists)
was followed by a weak positive gate and addition in §7. Its existence is not new in this
search. The [current catalog](../reference/masster_smarts_rules.md) is authoritative for the
22-rule set and measured gates. That gate measures peak coverage, not mechanistic proof.

The same audit's 14/15 AIMe example result concerns **reachable masses over exhaustive
connected atom subsets under the stated channels**. It does not prove correct product
connectivity, retention in the finite proposal pool, or performance across all chemistry.
Consequently it cannot establish that generic excision has no value, nor that its candidate
count would necessarily be prohibitive. Neither effect has been measured here.

A formula twin means an added product cannot expand exhaustive m/z support. It can still
change finite-budget selection or intensity features. Those are separate hypotheses requiring
matched experiments; the local zero-coverage results should not be generalized to all datasets.

## Recommended next checks

1. **N-phenylbenzamide O-transfer branch:** obtain the full 2015 reaction schemes, establish
   the atom map and named substituent restrictions, then count standardized training molecules.
   This is the clearest uncovered branch identified by this search.
2. **N-methylanilinoethanol:** resolve the previously capped connected-formula search on its
   existing molecules. Assess the morpholinylbenzoic-acid extension separately; do not fold it
   into a broad aminoalcohol pattern.
3. **Existing mapping backlog:** extract the sulfonylurea and benzyl-sulfone product schemes.
   Their prior corpus census is linked in the catalog; a new literature citation alone does
   not resolve the mapping gap.
4. **Optional low-priority census:** N-thioacyl sulfonamide transfer and PFAS sulfinate transfer.
   Retain only if the actual target corpus has relevant molecules and additional observed peaks.

For each proposed rule, use the existing
[per-class gate](../reference/masster_smarts_rules.md#rules-added-by-the-per-class-gate-2026-09-27):
atom balance, mode/scaffold restrictions, correct standardized tautomer, connected-formula
comparison, and a matched random-candidate baseline. Report separately the exhaustive support
and finite candidate-budget effects. Use training data for rule selection and reserve untouched
validation for confirmation; existing all-NIST20 exploratory gates are not held-out proof.
After a rule passes, measure candidate count, runtime, and spectrum accuracy at a matched budget.

No new coverage measurements or training runs were performed as part of this literature search. No complete AIMe supplementary
operator specification was obtained, so this is not an exact reproduction proposal.

## Subsequent coverage probe

The [continuation gate](../findings/masster/masster_excision_continuation_gate_2026_09_28.md)
subsequently tested generic bridge losses over c0 candidates on train and frozen validation
families. Aromatic nitro NO-loss continuations confirmed. A follow-up primary source,
[Pandeti et al., 2017](https://pubs.rsc.org/en/content/articlepdf/2017/ra/c7ra08312b), reports
side-chain cleavage followed by NO radical loss in protonated nitroimidazoles under ESI-CID.
This supports a specific additional family, not a general deletion operator. Coverage does
not establish atom mapping or a cosine benefit.
