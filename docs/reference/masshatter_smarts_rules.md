# MASShatter SMARTS Rule Catalog

> **Status:** Reference and research backlog
>
> **Updated:** 2026-08-22

This document defines which gas-phase rearrangement rules belong in MASShatter,
records the rules currently enabled, and keeps literature candidates separate from
validated code. It is intentionally narrower than the general
[rearrangement review](rearrangement_smarts.md).

## Scope and admission rule

MASShatter does not need every chemically valid SMARTS rule. Its ordinary mechanism
already enumerates bond-cut masks and the connected precursor components produced by
those cuts. A SMARTS rule is useful only when it adds a plausible atom partition that
the ordinary mechanism cannot represent for the corresponding cut mask.

Admit a rule when all of the following hold:

1. It creates a new heavy-atom assembly, normally by reconnecting retained atoms that
   are disconnected in the precursor after the neutral atoms are removed.
2. Its retained precursor atom set or partition is not already produced by ordinary
   MASShatter within the configured cut budget.
3. The precursor substructure can be expressed narrowly enough to avoid a large
   number of unsupported candidates.
4. The ion mode, adduct, scaffold, and substituent restrictions supported by the
   evidence can be encoded or otherwise enforced.
5. The retained and neutral atom maps conserve elemental composition apart from the
   model's explicit hydrogen-transfer channels.

Do not add a candidate rule merely because a rearrangement occurs. Proton transfers,
hydride shifts, resonance changes, and topology changes with the same ordinary atom
partition belong in ion-channel scoring or learned priors unless they demonstrably
recover missing candidate formulas.

## How MASShatter represents a rule

The curated registry is
[`mass_shatter_templates.py`](../../src/fragnnet/frag/mass_shatter_templates.py).
It selects shared definitions from
[`smarts_prepass.py`](../../src/fragnnet/frag/smarts_prepass.py) but owns a separate
allowlist.

- Rule index `0` is the ordinary connected-component mechanism.
- SMARTS rules receive stable one-based indices in allowlist order.
- A matched template stores the precursor bond cuts and an alternative partition of
  all precursor atoms.
- The cut mask keeps one probability; a SMARTS match adds another assembly mechanism
  under that mask rather than duplicating the mask probability.
- Multiple mechanisms that produce the same atom set remain latent alternatives.
- Literature-derived rules can restrict eligible precursor types and fragmentation
  modes; unsupported matches are removed before candidate construction.
- Generic `crf12`/`crf13` patterns and pure bond-deletion rules are intentionally
  excluded.

## Implemented allowlist

The current registry contains **26 rules: 12 compound-fragment rules and 14
neutral-loss rules**. These names are the executable source of truth.

### Compound-fragment rules

These rules perform several coordinated cuts and merge selected outer components into
a new assembly.

| Rule | Restricted scaffold | Candidate added |
|---|---|---|
| `anthraquinone_bis_co_loss` | Anthraquinone | Compound loss of both carbonyl oxygens and a flanking-ring C4 unit |
| `xanthone_ring_cleavage` | Xanthone | Reassembly of the two outer aromatic pieces after central-ring cleavage |
| `phenoxazine_ring_cleavage` | Phenoxazine | Reassembly of outer aromatic pieces across the N/O central ring |
| `furanocoumarin_ring_cleavage` | Linear furanocoumarin | Compound cleavage across the fused furan-benzene-pyranone system |
| `benzopyranone_ring_cleavage` | Coumarin/chromen-2-one | Reassembly following coordinated benzene and lactone-ring cuts |
| `halogenated_quinoline_ring_cleavage` | Halogenated quinoline | Ring fragment remaining after simultaneous N and benzo-carbon excision |
| `benzopyranone_bis_ch_loss` | Coumarin/chromen-2-one | Compound loss of one CH from each fused ring |
| `anthraquinone_ring_c4_plus_ch_loss` | Anthraquinone | Compound loss of one flanking ring plus one CH from the other ring |
| `morphinan_amine_arm_loss` | Oxide-bridged morphinan | Amine-arm loss compounded with a benzo CH loss |
| `desoxymorphinan_ring_cleavage` | Non-oxide morphinan | Reassembly across the tetracyclic ring system |
| `desoxymorphinan_aromatic_core_release` | Non-oxide morphinan | N-free aromatic-core candidate after amine-arm and CH2 loss |
| `pyridyl_carbamate_alkyl_transfer` | 2-, 3-, or 4-pyridyl O-carbamate | Transfer of Me, Et, or nBu to the pyridyl-O residue |

### Neutral-loss rules

Here, "neutral loss" means a mapped set of precursor atoms, not just a floating-point
mass. This distinction is essential when the retained atoms require a new assembly.

| Rule | Precursor class | Nominal transformation | Exact neutral mass (Da) |
|---|---|---|---:|
| `aryl_sulfonamide_so2` | Aryl sulfonamide | SO2 removal with new Ar-N assembly | 63.9619 |
| `aryl_sulfonate_so2` | Aryl sulfonate | SO2 removal with new Ar-O assembly | 63.9619 |
| `aryl_sulfonate_so3` | Aryl sulfonate | Direct SO3 loss | 79.9568 |
| `n_nitramine_n2o` | N-nitramine | N2O loss with retained-atom reassembly | 44.0010 |
| `sulfoxide_so_loss` | Dialkyl/diaryl sulfoxide | SO loss with new C-C assembly | 47.9669 |
| `diarylvinyl_c2h2_loss` | Acyclic diarylethylene | C2H2 loss with new aryl-aryl assembly | 26.0157 |
| `diaryl_ketone_co_loss` | Diaryl ketone | CO loss with new aryl-aryl assembly | 27.9949 |
| `aminobenzothiazole_hcn_loss` | 2-Aminobenzothiazole | HCN loss with ring reassembly | 27.0109 |
| `n_phenylbenzamide_co_smiles` | N-Phenylbenzamide | CO loss with new N-aryl assembly | 27.9949 |
| `methyl_benzoate_co2_transfer` | Aryl methyl ester | CO2 loss with methyl transfer to the aryl residue | 43.9898 |
| `anisoyl_fluoride_co_f_transfer` | o-, m-, or p-Anisoyl fluoride | CO loss with F transfer to the anisole ring | 27.9949 |
| `short_haloalkyl_ammonium_c2_transfer` | 2-Haloethyl amine/ammonium | C2H4 loss with Cl/Br/I transfer to N | 28.0313 |
| `short_haloalkyl_ammonium_c3_transfer` | 3-Halopropyl amine/ammonium | C3H6 loss with Cl/Br/I transfer to N | 42.0470 |
| `n_fluoro_sulfonamide_f_transfer` | N-Fluoroarenesulfonamide | Diagnostic SO2F- product with neutral ArN loss | Variable |

`aryl_sulfonate_so3` is the one implemented rule whose product atom set is an
ordinary connected remainder. Keep it under review: it is useful only if forcing its
cut mask or assigning its rule identity improves coverage or scoring beyond ordinary
enumeration.

## Literature additions implemented in 2026-08

These additions have a clear transferred heavy atom or group, a constrained substrate
class, and a product atom partition that adds coverage. They use the ionization guards
shown below. The short-chain halogen family requires separate C2 and C3 templates
because the leaving atom maps differ.

| Priority | Rule or family | Evidence-specific transformation | Eligibility | Why it fits MASShatter | Source |
|---:|---|---|---|---|---|
| 1 | `n_phenylbenzamide_co_smiles` | Deprotonated N-phenylbenzamide loses CO and forms a new N-aryl bond | `[M-H]-` | CO is internal to the path between retained groups | [Rapid Communications in Mass Spectrometry, 2015](https://pubmed.ncbi.nlm.nih.gov/26377014/) |
| 2 | `methyl_benzoate_co2_transfer` | Protonated aryl methyl ester loses CO2 while methyl becomes attached to the aryl residue | `[M+H]+` | Retains groups from opposite sides of the ester carbonyl | [Journal of Mass Spectrometry, 2018](https://pubmed.ncbi.nlm.nih.gov/29441643/) |
| 3 | `anisoyl_fluoride_co_f_transfer` | Anisoyl fluoride radical cation loses CO while F migrates from the acyl carbon to the ring | `[M]+`, EI | Explicit heavy-atom migration creates a new retained assembly | [Organic Mass Spectrometry, 1991](https://doi.org/10.1002/oms.1210261122) |
| 4 | `short_haloalkyl_ammonium_c2_transfer` and `short_haloalkyl_ammonium_c3_transfer` | Amine/ammonium chains of two or three methylenes lose C2H4/C3H6 and retain Cl, Br, or I on N | `[M+H]+` or `[M]+` | Strong chain-length and halogen guards make the transfer rule narrow | [Journal of the American Society for Mass Spectrometry, 2015](https://doi.org/10.1007/s13361-015-1261-4) |
| 5 | `pyridyl_carbamate_alkyl_transfer` | Protonated pyridyl dialkylcarbamates transfer Me, Et, or nBu to the pyridyl residue | `[M+H]+` | The carbamate linker is discarded while an alkyl group is retained remotely | [Tetrahedron, 2015](https://www.sciencedirect.com/science/article/pii/S004040201401518X) |
| 6 | `n_fluoro_sulfonamide_f_transfer` | Deprotonated ArSO2NF gives SO2F- plus the aryl-nitrogen remainder | `[M-H]-` | F and the sulfonyl group become a new charged assembly | [Rapid Communications in Mass Spectrometry, 2024](https://doi.org/10.1002/rcm.9722) |

The anisoyl rule is excluded when EI fragmentation-mode metadata is missing. The other
new rules are restricted by precursor type even when fragmentation mode is unavailable.
The general shared BFS registries remain unchanged.

## Verified ESI-CID expansion candidates

The following candidates were checked against primary reports that explicitly used
electrospray ionization followed by collisional activation. None of these candidates
requires EI metadata. They are not executable yet: each still needs an exact mapped
product, a coverage-delta check, and positive/negative SMARTS tests.

Priority `A` means the paper identifies the transferred heavy atom or group and the
proposed product appears to combine precursor atoms that ordinary bond deletion would
separate. Priority `B` means the chemistry is supported but the scope, multistage path,
or product mapping needs more work. Priority `C` requires metadata or graph handling
beyond a narrow small-molecule SMARTS.

| Priority | Proposed family | ESI eligibility | Rearranged candidate to encode | Remaining guard or mapping issue | Primary source |
|---|---|---|---|---|---|
| A | `n_methylanilinoethanol_o_smiles` | `[M-H]-`, CID or HCD | Phenoxide retains the aryl group and terminal alcohol O while the intervening N-alkyl chain leaves | Require N-alkyl substitution; the N-H analogue follows a different pathway and N-ethyl gives much less product | [JASMS, 2022](https://doi.org/10.1021/jasms.2c00210) |
| A | `n_acyl_aryl_sulfonamide_carbonyl_o_transfer` | `[M-H]-`, CID | Phenoxide contains the arylsulfonyl ring plus the oxygen transferred from the remote amide carbonyl | Keep N-acyl sulfonamides separate from primary sulfonamides and the existing SO2 rules | [JASMS, 2021](https://pubmed.ncbi.nlm.nih.gov/33586949/) |
| A | `diacylhydrazine_carboxylate_o_transfer` | `[M-H]-`, CID | `[R1CO2]-` is assembled by transferring the opposite amide oxygen to the R1 carbonyl carbon | Evidence is for `R1C(O)-N(t-Bu)NH-C(O)R2`; encode the tert-butyl and side-specific O map | [PLOS ONE, 2013](https://doi.org/10.1371/journal.pone.0063097) |
| A | `nitrobenzyl_hydroxybenzoate_co2_coupling` | `[M-H]-`, CID | CO2 loss is followed by coupling of the retained 4-nitrobenzyl and hydroxyaryl groups | Require a phenolic anion and the para-nitrobenzyl ester; validate vanillate substitutions separately | [JASMS, 2016](https://doi.org/10.1007/s13361-016-1339-7) |
| A | `proline_benzyl_ester_n_benzyl_transfer` | `[M+H]+`, CID | Benzyl transfers from the ester oxygen to the proline N while H2O and CO are lost | Separate the N-benzyl product from the competing benzyl-cation and hydride-transfer channels | [Journal of Mass Spectrometry, 2013](https://doi.org/10.1002/jms.3162) |
| A | `phenylalanine_carboxyl_o_transfer` | `[M+H]+`, CID/MS3 | After NH3 and CH2CO loss, carboxyl O is retained with the benzyl/aryl residue as a hydroxybenzyl cation | This is sequential and has competing benzyl-O versus ortho-aryl-O products with the same formula | [Frontiers in Chemistry, 2023](https://doi.org/10.3389/fchem.2022.1094329) |
| A | `primary_aryl_sulfonamide_o_transfer` | `[M+H]+`, CID/MS3 | A sulfonyl O transfers to the aryl ring to form a substituted phenoxonium ion | Product abundance depends strongly on ring substituents; require a supported conjugating substituent guard | [International Journal of Mass Spectrometry, 2019](https://doi.org/10.1016/j.ijms.2018.09.033) |
| B | `prazole_sulfone_so2_extraction` | Positive ESI, CID | Prazole sulfones lose the SO2 linker and form an N-(2-benzimidazolyl)pyridine framework | Restrict to sulfones; sulfoxides rarely form the product, thioethers require an additional SH-radical loss, and pyridine N-oxides are negative controls | [Molecules, 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10420917/) |
| B | `pyrimidylsulfanyl_benzamide_dual_smiles` | `[M-H]-`, CID | Two competing Smiles products retain either the dimethoxypyrimidyl side or the N-phenylamide side across the thiosalicyl linker | Requires two product templates and substituent-sensitive scoring | [JASMS, 2007](https://doi.org/10.1016/j.jasms.2007.07.012) |
| B | `sulfonylurea_herbicide_n_o_smiles` | `[M-H]-`, CID | Nitrogen- and oxygen-initiated Smiles channels produce alternative remotely assembled fragments | Evidence is strongest for rimsulfuron and nicosulfuron, not a generic sulfonylurea pattern | [Chinese Journal of Chemistry, 2012](https://doi.org/10.1002/cjoc.201200610) |
| B | `n_terminal_protected_amino_acid_o_migration` | Negative ESI, CID | A carboxyl oxygen participates through a five-membered transition state in protected Ser/Thr/Tyr derivatives | Exact retained atom maps must be extracted; activation depends on whether the C terminus is blocked | [Indian Journal of Biochemistry & Biophysics, 2006](https://pubmed.ncbi.nlm.nih.gov/17285801/) |
| B | `benzaldimine_hcn_benzyl_transfer` | `[M+H]+`, CID | Protonated N-benzylbenzaldimines lose HCN while benzyl transfer couples atoms from opposite sides of the imine | Start with the reported p-methoxybenzylidene series; ring electronics control transfer | [JASMS, 2014](https://doi.org/10.1007/s13361-014-0935-7) |
| B | `benzyl_phenyl_sulfone_benzyl_transfer` | `[M+H]+`, CID | Benzyl transfers across the sulfone before formation of characteristic product ions | The abstract establishes transfer across substituted analogues but the exact charged product map still requires full-scheme extraction | [Journal of Mass Spectrometry, 2021](https://pubmed.ncbi.nlm.nih.gov/33295108/) |
| B | `dibenzylaniline_benzene_loss_transfer` | `[M+H]+`, CID | A benzyl group transfers to a benzylic or aniline ring, followed by benzene loss | Two transfer destinations and several subsequent H shifts require separate atom-map validation | [Journal of Mass Spectrometry, 2017](https://pubmed.ncbi.nlm.nih.gov/28109035/) |
| B | `chromene_sulfonamide_sulfonyl_transfer` | `[M+H]+`, CID | Sulfonyl cation transfers from N to a phenyl carbon, then benzenesulfinic acid is lost with cyclization | Extremely narrow chromene-imine scaffold; electron-donating substituents favor the pathway | [Rapid Communications in Mass Spectrometry, 2016](https://pubmed.ncbi.nlm.nih.gov/26661975/) |
| B | `n_aryl_methacrylamide_co_claisen` | `[M+H]+`, CID | Amide Claisen rearrangement enables CO loss and a cyclized retained assembly | Electron-donating para substituents favor the reaction; product topology and atom map are multistep | [JASMS, 2012](https://pubmed.ncbi.nlm.nih.gov/23055072/) |
| B | `dichlorvos_phosphate_halogen_transfer` | `[M+H]+`, CID | P-O cleavage is followed by halogen-assisted transfer that forms protonated dimethyl halophosphate | Narrow organophosphate scaffold; Cl, Br, and F analogues were examined, but halogen and H maps must both be verified | [JASMS, 2017](https://doi.org/10.1007/s13361-017-1736-6) |
| B | `pfas_sulfonyl_f_transfer` | Negative ESI, CID/AIF | Perfluoroalkyl sulfonate or sulfinate loses CF2 and transfers F to give `FSO3-` or `FSO2-` | Keep sulfonates and sulfinates separate and restrict chain patterns to the studied PFAS series | [Rapid Communications in Mass Spectrometry, 2025](https://pubmed.ncbi.nlm.nih.gov/41539986/) |
| C | `protonated_glycan_fucose_transfer` | `[M+H]+` or `[M+NH4]+`, CID/in-source activation | A terminal fucose is retained on a remote glycan fragment, including inter-antenna transfer | Do not apply to `[M+Na]+` or `[M-H]-`; arbitrary glycan rings, multiple acceptor sites, and mobile-proton dependence make one SMARTS insufficient | [JASMS, 2002](https://pubmed.ncbi.nlm.nih.gov/11951970/), [Angewandte Chemie, 2018](https://pubmed.ncbi.nlm.nih.gov/29688603/) |
| C | `trimethyllysine_peptide_methyl_transfer` | Singly charged protonated peptide, CID | A methyl from trimethyllysine is retained on a remote C-terminal Arg fragment after backbone cleavage | Explicit precursor charge state and peptide-residue context are required; the transfer is absent for doubly charged ESI ions | [JASMS, 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2709603/) |
| C | `phosphopeptide_phosphate_relocation` | Positive or negative ESI, low-energy CID | A phosphate can relocate before backbone cleavage, yielding non-native phosphorylated sequence ions | Charge, basic residues, residue spacing, and activation method matter; HCD does not show the same slow-CID relocation | [International Journal of Mass Spectrometry, 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4375673/), [JASMS, 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5479408/) |

The first implementation batch should come from priority `A`, but not all seven rules
should be enabled at once. Measure corpus match counts and ordinary-mechanism coverage
for each rule independently before deciding which candidates justify model capacity.

## Additional literature backlog

These transformations are plausible MASShatter rules but need more spectra, narrower
guards, or a demonstrated coverage delta before implementation. "Provisional" means
the mechanism or scope is not yet strong enough to encode as a production rule.

| Proposed family | Transformation to investigate | Main restriction or concern | Source |
|---|---|---|---|
| `tosyl_imidamide_transfer` | Tosyl migrates to an anilinic N while phenylacetonitrile is lost | Limited imidamide series; substituent effects matter | [Journal of the American Society for Mass Spectrometry, 2015](https://pubs.acs.org/doi/10.1007/s13361-015-1156-4) |
| `catechol_phenethylamide_acylation` | Activated phenethylamides cyclize by internal acylation and then lose NH3 | Usually requires a highly activated, often dihydroxy aryl ring | [Journal of Mass Spectrometry, 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5555735/) |
| `aryl_phosphorothioate_s_transfer` | Protonated phosphorothioates ultimately form ArS+ after sequential methanol and phosphonate losses | Multi-step and likely needs pathway-specific scoring | [Journal of Mass Spectrometry, 2019](https://pubmed.ncbi.nlm.nih.gov/30569428/) |
| `dimethyl_diester_ketene_methoxy_transfer` | Dimethyl diesters lose ketene with methoxy migration | Old EI evidence and broad ester matching risk | [Journal of the Chemical Society C, 1968](https://doi.org/10.1039/J39680000202) |
| `benzoyl_carboxymethyl_transacylation` | Benzoyl migrates to a carboxyl group before benzoic-acid and CO losses | Narrow heteroaromatic scaffold and sequential mechanism | [Rapid Communications in Mass Spectrometry, 2009](https://pubmed.ncbi.nlm.nih.gov/19504483/) |
| `n_acyl_glycinamide_residue_loss` | N-acyl glycinamide loses the complete glycine residue to form a nitrilium-type ion | Confirm exact retained map and distinguish homologues | Literature lead; primary source still needs rule-level extraction |
| `n_acyl_beta_alaninamide_residue_loss` | N-acyl beta-alaninamide loses the complete beta-alanine residue | Homologue-specific chain guard required | Literature lead; primary source still needs rule-level extraction |
| `aryl_azo_n2_loss` | Aryl azo systems lose N2 and couple retained carbon centers | Charge state and substitution strongly affect pathway | Literature lead; primary source still needs rule-level extraction |
| `diaryl_sulfone_so2_loss` | Diaryl sulfone loses SO2 and joins the two aryl residues | Evidence is mainly EI; avoid broad sulfone matches | Literature lead; primary source still needs rule-level extraction |
| `heteroaryl_alpha_diketone_co_loss` | A heteroaryl alpha-diketone loses internal CO and reconnects retained groups | Evidence centered on 2-pyridyl systems | Literature lead; primary source still needs rule-level extraction |
| `aromatic_benzyl_ether_ch2o_loss` | Benzyl aryl ethers lose CH2O with retained-group reassembly | Must distinguish ordinary benzyl cleavage | Literature lead; primary source still needs rule-level extraction |
| `aminoacyl_benzylamine_formamide_loss` | Formamide loss transfers/reconnects the benzyl-bearing group | Narrow substrate family; map needs validation | Literature lead; primary source still needs rule-level extraction |
| `n_sulfonyl_phenylacetamide_hnco_loss` | HNCO loss accompanies benzyl-anion transfer | Negative-ion and substituent constraints needed | Literature lead; primary source still needs rule-level extraction |
| `meta_donor_benzamide_no_exchange` | N/O exchange gives a phenolic product plus benzonitrile | Requires meta electron donor; easy to overmatch | Literature lead; primary source still needs rule-level extraction |
| `benzyl_phenylalaninate_h2o_co_transfer` | Sequential H2O and CO loss retains a transferred benzyl group | Multi-step; exact charged product map required | Literature lead; primary source still needs rule-level extraction |
| `carbamate_co2_reassembly` | Carbamate CO2 loss reconnects the groups on either side | Too generic until the supported connectivity is resolved | Provisional mechanism |
| `polyfluoroaryl_cf2_ring_contraction` | Polyfluoroaryl ions lose CF2 with ring contraction | EI-only and specialized | Literature lead; primary source still needs rule-level extraction |
| `insti_hnco_benzyl_migration` | Integrase-inhibitor scaffold loses HNCO with benzyl migration | Provisional, drug-class-specific mechanism | Provisional mechanism |
| `tms_acetate_ch2co_migration` | Trimethylsilylacetic derivatives lose CH2CO with dimethylsilyl migration | Derivatization-specific EI rule | [Rapid Communications in Mass Spectrometry, 2002](https://pubmed.ncbi.nlm.nih.gov/11921371/) |
| `tms_steroid_a_ring_migration` | Trimethylsilylated cholesterol loses the A ring with TMS migration | Extremely scaffold- and derivatization-specific | [Journal of Mass Spectrometry, 2014](https://pubmed.ncbi.nlm.nih.gov/24234867/) |

Backlog entries without a linked primary source are research notes, not implementation
specifications. Locate and inspect the primary paper before turning any of them into
SMARTS.

## Excluded or scoring-only transformations

| Transformation | Decision | Reason |
|---|---|---|
| McLafferty and other proton/hydride rearrangements | Scoring or hydrogen-transfer channel | The heavy-atom subset is generally already reachable by ordinary cuts |
| N-benzylaniline benzyl-cation transfer | Exclude as candidate rule | The retained benzyl atom set is produced by ordinary bond cleavage despite the product topology change ([study](https://pubmed.ncbi.nlm.nih.gov/22823406/)) |
| Curtius/Wolff-type N2 loss | Exclude unless a coverage counterexample is found | The retained heavy-atom subset is normally an ordinary connected remainder ([study](https://pubmed.ncbi.nlm.nih.gov/40512019/)) |
| Metal-assisted AZT/phosphoramidate methoxy shifts | Exclude from the core registry | Strong metal/adduct dependence and often no new candidate mass ([study](https://pubmed.ncbi.nlm.nih.gov/15712358/)) |
| Remote fluorobenzyl transfer between amide nitrogens | Scoring-only by default | Transfer alone changes connectivity but not the precursor atom mask or mass ([study](https://pubmed.ncbi.nlm.nih.gov/22215578/)) |
| Generic cyclization, resonance, or charge migration | Exclude as candidate rule | Same retained atom set; use the model's mechanism and ion-channel scores |
| Pure single- or multi-bond deletion | Exclude from SMARTS | Already represented by ordinary MASShatter masks and connected components |
| Broad `crf12`/`crf13` patterns | Exclude | Match breadth creates unsupported candidates without a scaffold-specific mechanism |

## Validation gate for every new rule

A literature mechanism is not ready for the allowlist until it passes all of these
checks:

1. **SMARTS specificity:** positive examples match; close negative controls do not.
2. **Map correctness:** retained and neutral map numbers partition the intended
   precursor atoms with no omissions or overlap.
3. **Formula conservation:** calculated neutral composition and exact mass agree with
   the proposed mechanism; hydrogen differences stay within the configured channel
   range.
4. **Coverage delta:** the rule produces at least one target atom set or partition not
   available from the ordinary mechanism under the same candidate budget.
5. **Candidate load:** corpus-wide match counts and candidates per molecule remain
   acceptably small.
6. **Spectral evidence:** the added candidate explains reproducible peaks in held-out
   spectra, not only the paper's demonstration molecule.
7. **Ablation:** validation recall or likelihood improves without degrading common
   scaffolds or overwhelming the mechanism softmax.
8. **Tests:** add focused positive, negative, deduplication, partition, rule-index, and
   model-forward tests.

For a proposed rule, record the supported ion mode and adduct explicitly. If the
current template interface cannot enforce those restrictions, extend the interface
only alongside the first rule that requires them; do not add speculative generic
configuration.

## Maintenance

When the allowlist changes, update this document and verify the count against
`MASS_SHATTER_NUM_RULES`. Keep broad chemistry surveys in
[`rearrangement_smarts.md`](rearrangement_smarts.md); this file should remain the
decision record for MASShatter candidate generation.
