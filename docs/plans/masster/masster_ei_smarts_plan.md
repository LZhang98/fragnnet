# MASSter EI SMARTS rules: TMS silyl migration first

> **Status:** CLOSED NULL · **Updated:** 2026-10-18
> The NIST23 gate admits no EI rule ([gate](#nist23-gate-2026-10-18)). T1, the TMS Si–O–Si ion, has a new composition (reach 1.000) but adds −0.0018 nominal coverage, and its isotope test is borderline. The user declined the Fir arms (2026-10-18). The T1 code (off by default), the e6 configs and the Fir submitter stay in the repo (`f5728bdc`) and were never submitted.

Hub: [NIST23 EI](../../workstreams/nist23_ei.md). Related: [EI evaluation](../../findings/masster/masster_ei_evaluation.md)
(Metabolome m/z 147 composition and TMS-vs-non-TMS sections), [rewire rules plan](masster_rewire_rules_training_plan.md)
and [implementation spec](masster_rewire_rules_implementation.md), [SMARTS rule catalog](../../reference/masster_smarts_rules.md)
(admission rule and validation gate), and the EI SMARTS screen (CLOSED NULL; removed 2026-10-18;
`git show 95341fce:docs/findings/masster/masster_ei_smarts_screen_2026_10_01.md`).

## Objective and acceptance

Give the EI recipe (`e5`, `[M]+`, nominal targets with natural isotopes) candidates for EI rearrangement ions that no
connected atom set can make. Start with the TMS chemistry, where the measured gap is.

Accept a rule into the EI recipe only if all of these hold:

1. **Reach:** its product composition is unreachable by any connected atom set within ±4 H on the molecules it
   matches (catalog admission criterion 2, checked on compositions, not nominal masses).
2. **HR evidence:** on the HR-EI references, its products match measured intensity within 10 ppm that the
   baseline pool does not, and more than the topology null does.
3. **Trained effect:** in a matched arm, the ≥2-TMS strata improve on HR-EI (10 ppm) and NIST23 (1 Da). The gain
   must exceed the control seed spread, the null arm must not show it, and overall `val_cos_sim_1.0` must not drop
   by more than that spread.

## Evidence (established)

**The gap is real and Si₂-specific.** From the [EI evaluation](../../findings/masster/masster_ei_evaluation.md#why-recetox-metabolome-is-lowest):

- Metabolome measures m/z 147 at 147.0656 (C5H15OSi2⁺). It contains two Si and no backbone carbon, so bond cuts
  cannot make it.
- e5 instead puts that intensity on 147.0836 (C6H15O2Si⁺), which scores at 1 Da but misses at 10 ppm.
- Moving the predicted nominal-147 intensity to 147.0656 raises Metabolome cosine from 0.5726 to 0.5896.
- On NIST23 test, [TMS spectra lose −0.035 cosine](../../findings/masster/masster_ei_evaluation.md#tms-vs-non-tms) after
  mass reweighting, and the loss grows with TMS count and mass.

**Census, 2026-10-18** (this plan; e5 split `nist23_ei_inchikey_hr_ei_excluded_2026_10_17`, all three folds, 257,099
spectra). Artifacts: [`output/ei_tms_rule_census_2026_10_18/`](../../../output/ei_tms_rule_census_2026_10_18/)
(`tms_census.txt`, `tms_census_spec.pkl`, script `tms_census.py`, `inputs.md5`, `git_rev.txt`). A TMS group is
`[O,N,S][Si]([CH3])([CH3])[CH3]`, the same pattern the finding uses.

| TMS groups | Spectra (train / val / test) | 147 ≥10% of base | Mean L1 share at 147 | 191 ≥10% | 204 ≥10% | 217 ≥10% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 149,982 / 50,043 / 50,015 | 5.5% | 0.0036 | 3.4% | 2.4% | 2.3% |
| 1 | 2,193 / 695 / 696 | 7.8% | 0.0053 | 2.2% | 1.6% | 3.0% |
| 2 | 1,140 / 412 / 429 | 48.9% | 0.0419 | 5.1% | 3.2% | 6.5% |
| 3 | 449 / 134 / 165 | 63.2% | 0.0402 | 9.2% | 8.6% | 14.7% |
| ≥4 | 442 / 158 / 146 | 81.9% | 0.0505 | 20.4% | 28.0% | 57.4% |

- m/z 147 turns on with the second TMS group: 7.8% of 1-TMS spectra versus 48.9% of 2-TMS spectra. That is the
  signature of an ion that needs two Si.
- Validation has 704 spectra (605 molecules) with ≥2 TMS, and 576 spectra with ≥2 O-TMS.
- **Nominal NIST spectra carry the composition.** In the 2,053 spectra with ≥2 TMS and 147 ≥10% of base, the median
  148/147 ratio is 0.161. The natural M+1/M ratio is 0.158 for C5H15OSi2 and 0.118 for C6H15O2Si. The median
  149/147 is 0.122, above both compositions' M+2 (0.079, 0.043) because m/z 149 is also an ion of its own.
- A model with natural isotope envelopes can therefore learn the Si₂ composition from nominal training data.
- Other motif counts (spectra, all folds): TBDMS 2,300; nitro 12,217; sulfonyl 8,437; diaryl ether 1,737.

**Reachability, 2026-10-18** (brute-force connected-subgraph enumeration, ±4 H;
[`tms_rule_proto.py`](../../../output/ei_tms_rule_census_2026_10_18/tms_rule_proto.py), output `tms_rule_proto.txt`):

| Candidate ion | Molecule tested | Reachable by a connected set? |
| --- | --- | --- |
| 147 C5H15OSi2⁺ | glycerol 3TMS, 3-hydroxybutyrate 2TMS | **No** |
| 149 C4H13O2Si2⁺ | glycerol 3TMS | Only at the +4 H edge |
| 191 C7H19O2Si2⁺ | glycerol 3TMS | Yes (own-H −1, −2) |
| 204 C8H20O2Si2⁺• | 3-hydroxybutyrate 2TMS | Yes (own-H −2) |

217 and 204 in sugars are connected C2/C3 units, so bond cuts already reach them. That makes their low prediction an
intensity-allocation problem, not a candidate problem.

**Lessons from earlier nulls:**

- **Nominal coverage hides the gain.** The 2026-10-01 EI screen admitted nothing because, at nominal mass, every rule
  product collided with an ordinary channel or its isotope. T1 collides at nominal 147 in the same way, so a nominal
  coverage gate would reject it. Its gate is composition reach plus exact-mass HR evidence instead.
- **Rewire on ESI was flat.** On MassSpecGym, every rewire arm was within noise of the control, and `a0` equalled
  `a0null` on cosine ([rewire plan](masster_rewire_rules_training_plan.md#msgym-screen-2026-10-15)). The null arm is
  therefore mandatory here too.

## Literature basis

- **Si–O–Si migration ion and TMS transfer:**
  - [Harvey & Vouros 2019](https://doi.org/10.1002/mas.21590) (review).
  - [Byun & Gross 1997](https://pubmed.ncbi.nlm.nih.gov/9008870/): TMS⁺ and Me2Si=O migrate through an ion–neutral
    complex, even across rigid spacers.
  - [Rontani & Aubert 2004](https://doi.org/10.1002/rcm.1567): m/z 147/204/217 and chain-length dependence.
  - [Rontani & Aubert 2008](https://doi.org/10.1016/j.jasms.2007.10.014): hydroxy- and oxo-acid TMS transfers,
    checked with ²H labels.
- **Machine-usable rule sets for cross-checks:**
  - [Lai & Fiehn 2018](https://doi.org/10.1002/mas.21518): 228 TMS rules confirmed within 4 mDa.
  - [Wegner et al. 2014](https://doi.org/10.1021/ac403879d): 160 TMS/TBDMS fragment formulas.
- **Rearrangements physics-based simulation misses:** [Wang et al. 2022](https://doi.org/10.1021/acs.analchem.1c02838)
  (QCEIMS misses 37% of observed TMS ions).

## Rule catalog

All rules are rewire rules (`candidates/rewire_rules.py` format: mapped SMARTS plus delete/cut/form) with
`precursors=('[M]+',)`. TMS stands for `[Si]([CH3])([CH3])[CH3]`. Products below come from the prototype.

| ID | Name | Mapped SMARTS | Edit | Product | Status |
| --- | --- | --- | --- | --- | --- |
| T1 | `tms_siloxy_silylium` | `[#6,#15:1][OX2:2]TMS.[#6,#15:4][O,N:5][Si:6]([CH3:7])([CH3])[CH3]` | cut 1-2, 5-6, 6-7; form 2-6 | Me3Si–O⁺=SiMe2, C5H15OSi2⁺ 147.0656 at H shift 0 | **Build candidate** |
| T1b | `tms_siloxy_silanol` | `[#6,#15:1][OX2:2][Si:3]([CH3:8])([CH3])[CH3].[#6,#15:4][OX2:5][Si:6]([CH3:7])([CH3])[CH3]` | cut 1-2, 4-5, 6-7, 3-8; form 2-6 | Me2Si(OH)–O⁺=SiMe2, C4H13O2Si2⁺ at +1 H | Diagnostic only (G0-HR); reachable at +4 H; strong in RECETOX but weak in NIST23 (instrument effect suspected) |
| T2 | `vicinal_otms_geminal` | `[CX4;D2:1]([OX2]TMS)[CX4:3][OX2:4]TMS` | cut 3-4, 1-3; form 1-4 | (Me3SiO)2CH⁺ 191 at −1 H | Not built: composition reachable |
| T3 | `silyl_to_ester_carbonyl` | `[#6:9][CX4:8][CX3:1](=[OX1:2])[OX2]TMS.[OX2;!$(OC=O);!$(O[CX4]C=O):5][Si:6]([CH3])([CH3])[CH3]` | cut 8-9, 5-6; form 2-6 | [RCH=C(OTMS)2]⁺•, 204 for 3-, 4- and ω-hydroxy acids | Not built: composition reachable. The α-OTMS donor exclusion (`!$(O[CX4]C=O)`) is an untested refinement: without it the prototype gave 219 for 2-hydroxy acids, where the literature reports 190 |

**T1 donor scope.** The donor may be N-TMS as well as O-TMS: 2-TMS spectra with fewer than two O-TMS groups still
show 147 ≥10% of base in 39.1% of cases (568 spectra), against 63% for ≥2 O-TMS.

**T1 duplicate matches.** T1 matches each (acceptor, donor, donor-methyl) triple separately. Edit-distinct matches:
6 on each 2-TMS molecule tested, 18 on glycerol 3TMS (3 acceptors × 2 donors × 3 methyls), 60 on ribitol 5TMS. Every
match gives the same composition and m/z.
`masster_rewire_max_matches_per_rule: 8` keeps the first 8 in deterministic order, which loses no m/z support. G0b
reports the truncation and the extra rows. Add symmetry-class dedup only if G0b shows a cost problem.

**Backlog, not in this plan's first pass:**

- TBDMS analogues of T1.
- Steroid A-ring loss with 3-O-TMS migration, and the TMS-acetate dimethylsilyl migration (catalog backlog rows
  `tms_steroid_a_ring_migration` and `tms_acetate_ch2co_migration`).
- **Tier 2:** turning on `[M]+` for the existing ESI-admitted rewire rules with EI precedent (`sulfonyl_so2_extrusion`,
  `nitro_nitrite_NO_loss`, `diaryl_ether_o_excision`, the S- and CO2-extrusion rules). Each needs its own G0-reach and
  G0-HR, because nominal collisions killed the 2026-10-01 EI families.
- Azo N₂ loss and sulfonamide carbonyl-O transfer under EI are CLOSED NULL; do not re-propose them.

## Gates (model-free, before any training)

- **G0-reach (NIST23, all folds):**
  - Count T1 matches per molecule and fold.
  - For each matched molecule, check that C5H15OSi2 is unreachable by a connected set within ±4 H, using the pool
    from the 2026-10-01 screen (0–2 precursor-bond cuts, ±4 H).
  - Pass: unreachable on ≥95% of matched molecules. The expected exceptions are molecules that already contain
    Si–O–Si.
- **G0-HR (HR-EI references, 10 ppm, natural isotopes):**
  - Measure the share of measured intensity that T1 (and, as a diagnostic, T1b) products newly reach beyond the
    baseline pool, against the topology null (`rewire_null.py` shape), with a molecule bootstrap.
  - All TMS records are in Metabolome. The rule is fixed before this read and nothing is tuned on it.
  - Pass: the gain over the null has a 95% CI above zero.
- **G0b (cost):**
  - Matches, truncations, extras per spectrum against kept ordinary fragments (`K`), and node growth on
    ≥4-TMS molecules.
  - Wall-clock of a forward pass on a TMS-heavy batch against the control.
  - Non-TMS spectra must get zero extras, because T1 cannot match without Si.

## Build steps (after the user approves the plan; each step's tests run before the next)

1. **`[M]+` in the rewire stage:**
   - Add `"[M]+"` to `PRECURSOR_TYPES_ALLOWED`.
   - Append T1 at the END of `REWIRE_RULES` and `REWIRE_RULE_VOCABULARY` (append-only; existing ESI rules and indices
     unchanged).
   - Test: an `[M+H]+` spectrum of a TMS molecule gets no T1 match. Existing rewire tests stay green.
2. **EI channel check:**
   - Confirm that rewire extras go through the nominal-plus-natural-isotope channel path under `[M]+` (the stage was
     built and tested only on ESI adducts).
   - Pin in a test: on glycerol 3TMS the T1 extra's H-shift-0 channel is 147.0656, and its nominal envelope M+1/M is
     0.158 ± 0.001.
3. **Rule tests:**
   - Positives: glycerol 3TMS and glycine 3TMS (N-TMS donor).
   - Negatives: a 1-TMS molecule (cholesterol TMS), a TBDMS-only molecule, and a no-Si molecule.
   - Also test formula conservation, vocabulary hashes, and the null shape on the two-component pattern.
4. **Gate script** `scripts/debug/ei/debug_masster_ei_tms_rewire_gate.py`: G0-reach, G0-HR and G0b. Reuse the
   isotope and pool utilities of `debug_masster_ei_smarts_screen.py` and the reach logic of
   `debug_masster_rewire_rule_gate.py`.
5. **Configs** (only if G0 passes), under `config/masster/ei/`:
   - `e6_control`: the e5 recipe with a second seed. This is also the open "e5 seed replicate" thread.
   - `e6_tms`: as e6_control, plus `masster_rewire_rules: True` and `masster_rewire_continuation_cuts: 0`.
   - `e6_tms_null`: as e6_tms, plus `masster_rewire_null: True`.
   - All share one seed. Because only T1 lists `[M]+`, the rewire stage adds nothing to non-TMS spectra.

## Training arms and read (G1)

Same split, epochs and selection as e5 (`val_cos_sim_1.0`). Record the launch host and provenance at submission; the
HR-EI plan does not record where e5 ran. Read on validation first and use test once.

| Read | Metric | Pass |
| --- | --- | --- |
| HR-EI Metabolome, ≥2 TMS | 10 ppm cosine; share of predicted nominal-147 intensity on 147.0656 | `e6_tms` − `e6_control` gain above the \|e5 − `e6_control`\| difference on the same records, and `e6_tms_null` does not show it |
| NIST23 val, ≥2 TMS | 1 Da cosine and Stein, by TMS count | Same rule, with a molecule bootstrap |
| NIST23 val, all | `val_cos_sim_1.0` | No drop beyond the control seed spread |

If T1 passes, it goes into the EI recipe. The tier-2 rules then follow the same G0 → G1 path, one family at a time.

## Out of scope

- m/z 73 over-allocation: the mass is right, so this is intensity allocation (EI evaluation finding).
- The Metabolome-specific m/z 149: suspected to be an instrument (C-trap water) effect; a NIST-trained model is not
  expected to learn it.
- 204/217 intensity in sugars: these ions are reachable, so this is allocation.
- ESI use of T1: TMS derivatives are a GC-EI chemistry.

## Decisions (user, 2026-10-18)

1. **Gate on NIST23, not on the HR-EI references.** The model-free gates run on NIST23 EI (train and val peaks;
   test structures counted only), so RECETOX Metabolome stays a pure evaluation set. Because NIST23 is nominal, T1's
   evidence comes from composition reach, peak support against the topology null, and 148/147 isotope consistency,
   not from nominal added coverage.
2. **The e6 arms run on Fir.** The submitter must stage and md5-check the EI data, which may not be on Fir yet.
3. Tier-2 scope was not decided. The NIST23 gate evaluates the existing rewire rules under `[M]+` as well, as a
   measurement only. Production enables `[M]+` for T1 alone.

## NIST23 gate (2026-10-18)

Script [`debug_masster_ei_tms_rewire_gate.py`](../../../scripts/debug/ei/debug_masster_ei_tms_rewire_gate.py)
(tests `tests/test_debug_masster_ei_tms_rewire_gate.py`). Output
[`output/ei_tms_rewire_gate_2026_10_18/`](../../../output/ei_tms_rewire_gate_2026_10_18/): `summary.csv`, `rows.csv`,
`census.csv`, `t1_isotope_spectra.csv`, `t1_control_isotope_spectra.csv`, `provenance.json`. The full run (57,556
molecules, about 27 min, 8 workers) used no sampling; `..._sample20/` was a timing pilot.

Scope:

- Folds: train and val peaks scored; test structures counted only.
- Rules: T1, T1b, T2, T3, and the 17 committed ESI rewire rules evaluated as if they listed `[M]+`.
- Matches are truncated to 8 per rule, as in production. This truncates T1 on 813 of 2,067 molecules, which does not
  matter because every T1 product is the same C5H15OSi2.
- The null is 8 random-site topology draws per molecule and rule.

Decision rule: `b and (c or e)` with ≥20 val molecules.

- (b) is composition reach on all products, exact enumeration up to 40 heavy atoms.
- (c) is nominal added coverage with natural isotopes, against the null.
- (e) is the T1 148/147 isotope consistency. It needs its CI above 0.5 **and** above the upper bound of a 1-TMS
  control, where Si2 cannot form.
- Peak support (d) is descriptive only: a known TMS ion beats random null products by construction.
- The control and the descriptive (d) were added in the verification fix round, **after** the first result was seen.
  They are post-hoc.

| Rule | Molecules (train+val) | (b) unreachable | (c) rule − null, 95% CI | Gate |
| --- | ---: | ---: | --- | --- |
| T1 `tms_siloxy_silylium` | 2,067 | 1.000 | −0.0018 [−0.0029, −0.0007] | Fails on train+val (e below), borderline |
| T1b `tms_siloxy_silanol` | 1,826 | 0.801 | −0.0094 [−0.0107, −0.0080] | Fails (b, c) |
| T2 `vicinal_otms_geminal` | 180 | 0.017 | +0.0024 [+0.0011, +0.0037] | Fails (b) |
| T3 `silyl_to_ester_carbonyl` | 227 | 0.379 | +0.0024 [+0.0003, +0.0047] | Fails (b) |
| Best ESI rules under `[M]+` | `sulfonyl_so2_extrusion` 6,677; `s_to_o_aromatic_C` 5,301; `thiono_thiolo_aryl` 119 | 0.835; 0.754; 0.916 | +0.0071; +0.0061; +0.0050 (all CIs above 0) | Fail (b); no ESI rule reaches 0.95 |

T1 isotope consistency (e), share of spectra where the observed 148/147 is closer to C5H15OSi2 than to the best
reachable nominal-147 composition (exhaustive competitors, molecule bootstrap):

| Fold | T1 (≥2 TMS, 147 ≥10% of base) | 1-TMS control | Median abs. error, Si2 vs competitor |
| --- | --- | --- | --- |
| train+val | 0.739 [0.714, 0.764] (1,164 mol / 1,453 spectra) | 0.667 [0.597, 0.731] (201 / 221) | 0.0115 vs 0.0231 |
| val | 0.780 [0.731, 0.826] (287 / 368) | 0.566 [0.434, 0.698] (53 / 60) | 0.011 vs 0.023 |

Reading:

- **T1's composition is new, but nominal NIST23 cannot prove it.** In nominal mode its 147 adds no coverage (the
  ordinary C6H15O2Si sits at the same nominal mass), so (c) is slightly negative.
- **The isotope evidence is moderate.** The Si2 ratio fits about twice as well as the competitor (median error 0.011 vs
  0.023). It beats the control on val, but not on train+val, where the intervals touch (0.714 vs 0.731).
- **The control is weak.** It has 221 spectra, and its median error is about 0.10 for both compositions.
- **Under the rule as written, T1 is not admitted.** The direct evidence for its composition remains the HR-EI exact
  mass (Metabolome 147.0656), which by decision 1 was not used as a gate.
- **No other candidate passes reach.** T2/T3 (and mostly T1b) were expected to fail. The ESI rewire rules add some
  nominal coverage under `[M]+`, but 9-25% of their matched molecules already reach the product composition, so
  none is admitted for EI.

Shared-anchor gap in the committed T1 SMARTS: 431 of 3,162 NIST23 molecules with ≥2 O/N-TMS groups get no T1 match.
These are 90 shared-anchor O-TMS pairs (for example tris-TMS phosphate and geminal bis-OTMS), 119 N,N-bis(TMS) amines,
and the rest have no O donor. The SMARTS is in the CRC32 hash, so fix it, or accept the gap, before the first Fir arm.

## Progress

- **2026-10-18:** Literature pass, census, isotope-ratio check, reachability prototype and rule catalog (this
  document).
- **2026-10-18:** Built T1 as rewire rule 18 (`[M]+` only, off by default), with tests, the e6 configs and
  `scripts/manual_slurm/fir/submit_masster_ei_tms.sh`; committed in `f5728bdc`. Ran the NIST23 gate (above):
  no rule admitted. **User decision: T1 does not go to Fir.** Closed.
- **To revive:** use exact-mass evidence. The HR-EI references (Metabolome 147.0656), or HR-EI training data, are
  the only data that can separate C5H15OSi2 from C6H15O2Si; nominal NIST23 cannot. Freeze the T1 SMARTS
  (shared-anchor gap above) before any arm trains.
