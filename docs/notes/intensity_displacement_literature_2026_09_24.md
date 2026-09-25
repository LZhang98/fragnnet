# Intensity displacement: what the literature says about our Phase 0 results

> **Status:** NOTE · **Updated:** 2026-09-24
> A literature search for the
> [intensity-displacement plan](../plans/masster/masster_intensity_displacement_plan.md),
> read against its [Phase 0 finding](../findings/masster/masster_intensity_displacement_phase0_2026_09_24.md).
> It covers three areas: forward MS/MS predictors, fragmentation energetics, and proteomics
> intensity models. §7 is a second pass that corrects two first-pass claims. Generic
> memorization and regularization work is in the
> [memorization note](memorization_literature_2026_09_24.md) and is not repeated here. Several
> primary papers were paywalled; claims taken from secondary summaries are marked
> *(unverified)*.
>
> **In short:** the literature explains the low-CE fallback and points to one concrete design:
> tie precursor survival to per-bond break tendencies with a CE-dependent breakdown curve. It
> independently confirms that neutral-loss identity helps generalization. It adds one data-side
> check we have not run (collision-energy calibration per instrument). It does **not** support
> ranking losses, retrieval for the novel tail, or receptive-field restriction as validated
> prior art.

## What Phase 0 established

On unmerged `[M+H]+`/FT:

- About two thirds of the val intensity misplacement is bias shared by independent runs.
- It is misordering among real peaks, not confusion with neighbouring fragments.
- The one structured part is a precursor/parent over-hold below NCE 60.
- There, the lost neutral's formula stacks with the model for +0.03 pairwise AUC.

The question for the literature: what explains the fallback, what representation transfers
lability to new molecules, and is any of the shared bias data-side rather than model-side?

## 1. The fallback is what unimolecular kinetics predicts

- **Mechanism.** In RRKM/QET competition, rearrangement losses through tight cyclic transition
  states often have lower critical energies E0 than simple cleavages, but smaller activation
  entropy. Near threshold (low CE) they are the only open channels, so they win. Loose-TS simple
  cleavages take over only at higher internal energy
  ([tight vs loose TS](https://www.sciencedirect.com/science/article/abs/pii/000926149190456J)).
  The losses that beat the precursor in the dominant-peak failing cohort (H₂O, CO₂, NH₃,
  glycosidic, HCl) are largely of this rearrangement type. So precursor survival at low CE is set
  by whether the molecule has **one low-barrier channel**, which is a local property. Our model
  instead falls back to a population prior on unseen molecules.
- **CFM-ID models this coupling directly.** Survival mass is an outcome of the same per-bond break
  tendencies that produce the fragments
  ([CFM-ID](https://arxiv.org/abs/1312.0264), [CFM-ID 4.0](https://academic.oup.com/nar/article/50/W1/W165/6591530)).
  FIORA does similar: a precursor-stability logit competes with per-bond fragment logits in one
  softmax ([FIORA, bioRxiv full text](https://www.biorxiv.org/content/10.1101/2024.04.22.590551.full.pdf),
  [Nat. Commun. 2025](https://www.nature.com/articles/s41467-025-57422-4)).
- **Where MASSter differs.** Its precursor is a candidate token whose score is a free residual
  of the global transformer. With `proposal_selection_only: true`, nothing ties that score to the
  lability of the molecule's bonds.
- **Nobody fits a continuous breakdown curve.**
  - CFM-ID trains discrete low/medium/high energy models.
  - FIORA, ICEBERG and GLACIER feed CE in as a covariate or embedding.
  - ICEBERG masks the precursor peak at evaluation
    ([ICEBERG-2](https://www.biorxiv.org/content/10.1101/2025.05.28.656653v1.full)).

  A per-bond curve, `p_break(bond, CE) = σ(slope_b · (g(CE) − threshold_b))`, is the parametric
  form QET suggests. It has a threshold, standing in for E0, and a slope, standing in for how
  loose the transition state is. It appears unexplored.

**Lesson for 2A.** The plan's local prior uses the release log-probability under per-bond scores.
For the precursor, that is `Σ log(1 − p_break)` over all bonds, which is exactly a CFM-style
survival term. Two changes follow:

1. make the bond scores CE-dependent through a threshold/slope breakdown curve, instead of an
   unconditioned score;
2. keep the precursor inside the same prior rather than as a free token.

The Phase 1.4 test stays the gate: a bond-local learner must beat the formula stack on
precursor-versus-contender.

## 2. Neutral-loss identity carries transferable signal (independent confirmation)

- **GRAFF-MS** predicts over a formula vocabulary in which each product ion can equally be
  represented by its neutral-loss formula. The authors state that this symmetry "is crucial to
  generalization": unseen, heavier molecules rarely share exact product-ion formulas with the
  training set, but they share a small recurring vocabulary of losses
  ([arXiv 2301.11419](https://arxiv.org/pdf/2301.11419)). This matches our stacking result from
  a different group and architecture.
- **Where the lesson applies here.** The MassSpecGym GLACIER-parity arm (`..._glacier_feats.yml`,
  prepared, not submitted) adds `c_nl_composition`, the complement's element counts. That is the
  input form of GRAFF-MS's lesson. The
  [formula-only baseline](../findings/masster/masster_formula_only_baseline_2026_09_23.md)
  found formula information adds +0.005 overall. Phase 0 says its value is concentrated in the
  low-CE precursor decision. Read that arm on the low-NCE precursor split specifically, not only
  on mean cosine.
- **The output form is untested.** GRAFF-MS's output, a distribution over loss formulas, is a
  different parameterization. It is lower priority, because the whole global-softmax output
  family has repeatedly nulled here.

## 3. A data-side source of shared bias we have not checked: CE calibration

- **Proteomics calibrates NCE per dataset.** Prosit and Oktoberfest sweep about 30 candidate NCE
  values on confident spectra and keep the one with the highest spectral similarity
  ([Oktoberfest](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/pmic.202300112),
  [docs](https://oktoberfest.readthedocs.io/)). Effective NCE differs across instruments.
- **UniSpec** reports that raw eV input beats NCE by 6-9% similarity
  ([bioRxiv](https://www.biorxiv.org/content/10.1101/2023.06.14.544947v1.full)).
- **Why this matters here.** A per-instrument CE offset would appear exactly as *shared* bias,
  since both runs see the same mislabeled CE. It would be worst where CE matters most, at the
  precursor/first-loss decision.
- **What blocks the check.** NIST20 FT spectra come from several Orbitrap models, but `spec_df`
  keeps only `inst_type` (FT), and the unmerged `[M+H]+`/FT configs use NCE only (`nce_only`,
  `ace: false`). The raw NIST20 MSP is not on this workstation.
- **The check:**
  1. Join the raw MSP `Instrument` field to `spec_id` on the cluster.
  2. Stratify the Phase 0 precursor over-hold (model minus target precursor share, NCE ≤ 30) by
     instrument model at matched NCE.
  3. If one instrument shows a consistent offset, fit a Prosit-style per-instrument NCE shift on
     train, or run a local NCE sweep at inference on a val subset.

  This separates CE mis-registration from missing lability knowledge, the plan's H1.

## 4. Local versus global: weaker prior art than it looks

- **FIORA's generalization claim is narrow.** It computes each bond's logits from the two
  endpoint atoms after 6 message-passing layers and normalizes globally. It degrades less than
  ICEBERG from high- to low-similarity bins: 18% against 26% relative cosine loss (primary text).
- **But the comparison is not local against global.** ICEBERG is also a bounded message-passing
  model, so the test is local against local, confounded with single-step versus autoregressive
  scoring. Six hops also covers most of a small molecule.
- **GLACIER, the most globally attentive model surveyed**, posts the best scaffold and
  MassSpecGym numbers
  ([arXiv 2606.29161](https://arxiv.org/html/2606.29161v1); [our note](glacier_object_detection.md)).
- **Consequence.** Keep 2A's identity-blind path **additive**, next to the global head, not a
  replacement. The [route map](../findings/masster/masster_route_map_2026_09_24.md) finding
  that global attention writes the memorization key argues for a separate identity-blind path.
  The literature does not show that removing global attention helps.
- **Proteomics is consistent with this.** Every strongly generalizing peptide model computes
  intensities from position-local representations: Prosit's BiGRU with per-site decoding;
  UniSpec's per-position transformer, with a train/test cosine gap of about 0.03; DeepGlyco's
  tree-LSTM over glycan topology. The Prosit Transformer gained when attention was put in the
  decoder, not the encoder. Peptides are far more local than small molecules, so this is
  corroboration, not evidence.

## 5. Things the literature argues against

- **Ranking or listwise losses.** No controlled ablation shows a ranking loss improves held-out
  peak order in any spectrum model. Proteomics answered ordering and calibration problems by
  reformulating the target instead (presence probability: Pep2Prob, Alpha-Frag). This agrees
  with our train ranking gap of 0.011.
- **Retrieval augmentation for the novel tail.** MARASON gains cosine 0.727 → 0.765 (random
  split) and 0.699 → 0.725 (scaffold), but "RAG starts to bring a significant performance gain
  when the Tanimoto similarity is larger than 0.3"
  ([arXiv 2502.17874](https://arxiv.org/html/2502.17874)). Our worst bins are below 0.4, so the
  gain lands where we are already best.
- **Proton affinity as a lability prior.** An ab initio MD study on protonated ions, matched
  against NIST20, found that calculated protonation-site PA predicts outcome weakly, and that the
  highest-PA protomer sometimes matched worst
  ([CIDMD, JCIM 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11492807/)).
- **Homolytic BDE predictors** (ALFABET) model neutral homolysis, the wrong regime for
  even-electron `[M+H]+`. BonDNet handles heterolytic and charged cases, but only for small
  C/H/O/F/Li chemistry
  ([ALFABET](https://www.nature.com/articles/s41467-020-16201-z), [BonDNet](https://pubs.rsc.org/en/content/articlelanding/2021/sc/d0sc05251e)).
  ICEBERG-2 reports that its learned cleavage order "associate[s] with DFT calculation of
  inductive cleavage energy", on two case-study molecules and with no correlation reported
  (verified in the bioRxiv text; see §7). That is a possible diagnostic, not a feature to add.
- **QM simulation at training scale.** QCxMS2 reaches entropy similarity 0.687 (HCD) / 0.773
  (CID) on its own evaluation set, and costs far too much for 12k molecules
  ([ChemRxiv](https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-gcws2)). No one has distilled
  it into an ML predictor. At most it is a diagnostic on a few hundred failure cases.

## 6. Minor

- **Dynamic range.** UniSpec trains on square-root intensities, and Pep2Prob is motivated by
  intensity models being over-sharp. Both match our over-sharpness (entropy −0.15 nats, `P^0.75`
  +0.007). The template already has `ints_transform: sqrt`. An older FraGNNet note found sqrt
  training hurt retrieval ([pairwise loss](../reference/pairwise_loss.md)). The memorization
  note's label smoothing aims at the same symptom. Low priority.
- **CE50 is structure-specific, not a mass effect.** *Corrected in the second pass:* Kertesz et
  al. found CE50 "orthogonal to compound mass". It discriminated structural and positional
  isomers and correlated with Molconn structural descriptors over 54 compounds
  ([JASMS 2009](https://doi.org/10.1016/j.jasms.2009.06.002)). This supports the premise that
  survival is a molecule-specific structural property. It fits our formula-only learner, which
  sees mass and NCE, not beating the model alone.

## Ranked next steps for the plan

1. **Per-instrument CE-offset check** (§3). Offline once the instrument field is joined, and it
   is the only lever here that is data-side. If positive, a per-instrument NCE calibration is a
   config-level fix for part of the shared bias.
2. **Phase 1.4 as planned, with the CFM framing** (§1). A bond-local learner on
   precursor-versus-contender: does bond-local structure add AUC beyond the formula stack? If it
   does, build 2A with a CE-dependent threshold/slope bond curve and the precursor inside the
   prior.
3. **Read the prepared GLACIER-parity arm (`c_nl_composition`) on the low-NCE precursor
   decision** (§2) when it runs. Mean cosine alone would hide it.
4. **Not pursued:** ranking losses, retrieval, proton-affinity or homolytic-BDE features, QM
   distillation at scale, and removing global attention as a displacement fix.

## 7. Second pass (2026-09-24): corrections and additions

Searched through PubMed and Consensus, with primary text checked where noted.

**Corrections to the first pass**

- **Proton transport is not the barrier for small molecules.** Wright et al. stepped CE from 5
  to 50 eV on seven small molecules. Proton migration was "independent of collision energy", and
  the proton "appears to be fully mobile at 5 eV". They conclude that CE drives bond vibrational
  energy and cleavage *rates* within the collision-cell residence time
  ([RCM 2016](https://doi.org/10.1002/rcm.7521)). This removes the first pass's "CE partly pays
  for proton delivery" reading and leaves the per-bond kinetic picture of §1.
- **CE50 is orthogonal to mass** (Kertesz 2009); see §6.

**Support for per-bond, CE-continuous curves (§1)**

- **Tight versus loose transition states are measured for a protonated molecule.** Guided-ion-beam
  thresholds for protonated tetraglycine were modelled with RRKM. They put water loss on tight
  transition states and the b-ion cleavages on loose ones
  ([Perez et al., J. Phys. Chem. B 2025](https://consensus.app/papers/details/96b8a70811635ced8cf445f023f87623/?utm_source=claude_desktop)).
- **Intensity is smooth in CE.**
  - Spectra at unreported CEs can be interpolated accurately from as few as three measured
    spectra ([King et al., JCIM 2022](https://consensus.app/papers/details/bd87f42be5c457c4930dcf01a3c6bf6c/?utm_source=claude_desktop)).
  - METLIN 960K fits a spline to each transition's CE-dependent intensity from four energies (0,
    10, 20, 40 eV) to predict optimal CE
    ([Aisporna et al., Anal. Chem. 2025](https://doi.org/10.1021/acs.analchem.5c04639)).

  So a few parameters per fragment describe the CE response, which argues for a
  threshold-and-slope bond curve over a free per-CE embedding.
- **ICEBERG-2's CE behaviour matches the kinetic picture** (bioRxiv text). C–O and C=O cleavages
  matter most at low CE. Aromatic bonds break only toward 100 eV. It encodes CE in eV with a
  64-dimensional sinusoidal embedding into both stages
  ([bioRxiv 2025.05.28.656653](https://www.biorxiv.org/content/10.1101/2025.05.28.656653v1.full)).

**Cross-instrument energy scales (§3)**

- **HCD internal energy is linear in collision voltage for each instrument.** An RRKM
  thermometer-ion study on Orbitrap HCD found this and proposed voltage-based calibration
  ([Ichou et al., JMS 2014](https://consensus.app/papers/details/5706e7a969c950c7ac941d0bd64de6f9/?utm_source=claude_desktop)).
- **Instrument-specific lab energies can be aligned by breakdown-curve features.** Hueber et al.
  used "singular points" of ERMS breakdown curves to find where spectra from different
  instruments become nearly identical
  ([JASMS 2024](https://consensus.app/papers/details/82eededd5ea95e8a8ee8dc6fd1f8f7f6/?utm_source=claude_desktop)).
- **Cross-vendor agreement depends on CE.** Across five vendors, spectra of 100 standards agree
  best at 20 eV and significantly less at 0, 10 and 40 eV
  ([Hoang et al., Anal. Chem. 2024](https://consensus.app/papers/details/1207e850f6185699a56d7502f69877ec/?utm_source=claude_desktop)). Orbitrap NCE
  30-60 matched QTOF 20-50 eV in one two-library comparison
  ([Oberacher et al., Metabolites 2018](https://consensus.app/papers/details/10e8371d2326511e9d7f2f61a24fa41a/?utm_source=claude_desktop)).

These make instrument-dependent effective energy well documented for small molecules. The §3
per-instrument check stays first.

**The high-CE part, which Phase 0 left unexplained**

- **Radical fragment ions are common but not our high-CE driver.** 65.4% of positive-mode NIST20
  spectra from even-electron precursors hold at least 10% radical fragment ions by count,
  especially aromatic and conjugated compounds, with CE-dependent trends
  ([Xing et al., Anal. Chim. Acta 2022](https://consensus.app/papers/details/5f29cbfd1d2d52e1a512675cb33ec3dd/?utm_source=claude_desktop)). In our classes, radical and even-electron partners fall under H shift. Checked
  on the Phase 0 support-null sample, H-shift displacement is at chance in every NCE bin:
  0.98x (≤ 20), 0.82x (20-40), 1.12x (40-60), 0.89x (> 60).
- **Consecutive dissociation.** HCD spectra differ from ion-trap CID because of multiple
  dissociation events, which reach more endothermic and consecutive pathways
  ([RDD-HCD, JASMS 2023](https://consensus.app/papers/details/47acc801f7cf560bb982e6c5c50d1b84/?utm_source=claude_desktop);
  [saccharide HCD, JMS 2018](https://consensus.app/papers/details/59c701b75f9f596a95e3f957f7d4700f/?utm_source=claude_desktop)). A plausible reading of the high-CE misordering is that the
  winner is set by the *stability of intermediate fragments against further decomposition*,
  which a single softmax over candidates does not represent mechanistically. **Caution:**
  FraGNNet's conserved DAG-flow output, the direct test of sequential mass flow, trained
  null-to-harmful ([FluxMPNN plan](../plans/fragnnet/fluxmpnn_plan.md)). Treat this as a
  hypothesis with a prior null, not a lead.

**New models, low relevance**

- **HDSE-MS** (Anal. Chem. 2025) adds hierarchical distance encoding to a transformer and reports
  entropy similarity 0.759 on NIST23 `[M+H]+`
  ([doi](https://doi.org/10.1021/acs.analchem.5c04111)).
- **SingleFrag** (Brief. Bioinform. 2025) predicts only whether each of the 1,000 most common
  0.01 m/z bins is present, merged across CE. It drops intensity and CE, so it has nothing on
  ordering ([doi](https://doi.org/10.1093/bib/bbaf333)).

**Net effect on the ranked steps:** unchanged in order. §1's per-bond CE curve has stronger
support; §3's check has more precedent; and the high-CE residual has one mechanistic hypothesis,
with a prior null against its most direct test.
