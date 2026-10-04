# SMARTS usage audit (2026-09-23)

> **Status:** FINDING · **Updated:** 2026-09-23 · Read-only audit of every SMARTS path, from
> rule definition to model input, plus a model-free census on NIST20. No code, config, DAG or
> data changed; nothing trained. Probe: `scripts/debug/smarts/debug_smarts_rule_census.py`.

## Answer

1. **No production model uses a SMARTS rule today.** MASSter's templates are off and
   cannot be turned on in any current arm, and production FraGNNet trains on the no-prepass
   DAG (§1).
2. **The registry holds two different kinds of rule, and only one of them is new chemistry
   for MASSter.**
   - The eleven compound ring-cut rules shared with FraGNNet cut existing bonds. Their
     intended ion is an ordinary fragment of the cut mask. The only piece they add in
     MASSter is the *merged neutral byproduct*, which explains intensity at the chance
     floor.
   - The neutral-loss rules (and `pyridyl_carbamate_alkyl_transfer`) produce retained atom
     sets that no cut can. Seven of them explain real intensity well above that floor (§2).
3. **The measured ceiling for the existing new-assembly rules is small.** Summed over
   `[M+H]+`/`[M-H]-`, their products mass-match about **0.1% of NIST20 intensity**,
   concentrated in 1,429 molecules (4.1%). It is large per molecule (aminobenzothiazole
   `[M-H]-`: 27% of a spectrum). SMARTS is a coverage device, and coverage is not what
   limits either model on this split
   ([overfit read-out](masster/masster_overfit_readout_2026_09_22.md)).
4. **Four defects, one of them silent**, plus two provenance traps in the DAG stats that
   past comparisons relied on (§3, §4).

## 1. Where SMARTS reach a model

| Path | Switch | Production state |
| --- | --- | --- |
| MASSter templates → `SmartsMechanismHead` (rule embedding in the per-mask mechanism softmax) | `masster_smarts_templates`, default `False` (`config/template.yml:775`) | **Off everywhere.** On only in the two `config/masster/merged/smarts/` configs, which never ran on real data (last handoff item: "Next: a real-data GPU smoke check"). |
| MASSter CUDA candidate backend | `masster_candidate_backend: cuda` (103 recent configs, incl. `c4a_base_s1`) | Refuses SMARTS at construction (`masster_model.py:1398-1411`, `masster_cuda_full_candidates.py:182`). |
| MASSter `fragment_selection: marginal`, `connected_fragment_beam` | new 2026-09-23 selection | Also refuse SMARTS (`masster_model.py:1351-1396`). |
| MASSter standalone inference | `script_utils` uses `to_device`, not the PL hook | Silently drops SMARTS mechanisms ([code audit §1](masster/masster_code_audit.md#1-standalone-inference-silently-omits-smarts-mechanisms)); still open. |
| FraGNNet DAG prepass (`FRAG_RULES` + `NL_RULES`, all or nothing) | `smarts_prepass` at DAG build → `..._smpTrue` DAGs | Production configs (`fraggnn_d3_ma_mi_nist20_amolc_all.yml`, all four `ce_pair_2026_09_23` arms) read `..._smpFalse`. Only `new_dags/..._prepass.yml` and `d3_new_features/..._prepass_new_mol_feats.yml` (`alvhehe2`, lost) read `smpTrue`. |
| FraGNNet node features | — | No rule identity reaches the model (`prepass_rule_id` removed 2026-04-14). The only prepass signal is the depth one-hot, which is wrong for prepass nodes while `num_cut_bonds` is off ([feature headroom](feature_headroom_2026_09_22.md)). |
| Atom/bond/molecule features | — | No SMARTS-based feature in `feat_utils.py`; MACCS keys are NEIMS-only. |

Past the proposal probability, the MASSter rule index goes nowhere. It is not a candidate
feature or an intensity-token input, and nothing logs template firing or the intensity
template candidates explain.

## 2. What each rule adds

### 2.1 Census: the broad rules are the FraGNNet prepass

Production matcher (`GetSubstructMatches`, default `uniquify=True`) over the 35,025
`mol_df` molecules:

| Rules | Molecules matched | Partitions yielded |
| --- | ---: | ---: |
| `crf12_0`, `crf12_1`, `crf13_2` (FraGNNet only) | 27,989 / 9,093 / 7,546 | 450,316 / 22,527 / 11,330 |
| All 29 other rules combined | — | ≈ 5,000 |

The broad rules yield **99.0%** of FraGNNet prepass partitions. So the `alvhehe2` verdict
("the prepass adds noise") and the +0.0006 oracle cos
([DAG oracle ceiling](fragnnet/dag_oracle_ceiling_nist20.md)) are measurements of `crf12`/`crf13`.
**A curated-rules-only DAG has never been built or scored.**

Dead or near-dead rules:

- `sulfoxide_so_loss`: 0 of the 120 sulfoxide molecules. The standardizer writes
  `[S+][O-]` and the pattern needs `S=O`.
- `anisoyl_fluoride_co_f_transfer`: 0 molecules, and it requires EI and `[M]+`, which never
  occur.
- `n_fluoro_sulfonamide_f_transfer`: 0 molecules.
- `pyridyl_carbamate_alkyl_transfer`: 1 molecule.
- `phenoxazine_ring_cleavage`: 4 molecules.

### 2.2 Which side of a template is new, and does it explain peaks

Each MASSter template has two components, split below into two groups:

- **ordinary**: the component that the same bond cut already produces as a connected
  component;
- **template-only**: the component that exists only through the template.

"Explained" is the spectrum intensity mass-matched at 0.01 Da, allowing a −6..+4 H window
(method in the probe docstring). The two numbers per cell are the mean fraction and the share
of spectra above 1%.

**Compound ring-cut rules.** In every one of the eleven, the template-only side is the merged
neutral byproduct. Examples:

- `anthraquinone_bis_co_loss`: C4O2, i.e. both carbonyl O plus a C4 unit;
- `xanthone_ring_cleavage`: C8, i.e. two unbonded C4 units;
- `halogenated_quinoline_ring_cleavage`: CClN.

| Rule | Template-only `[M+H]+` | Template-only `[M-H]-` | Ordinary `[M+H]+` | Ordinary `[M-H]-` |
| --- | --- | --- | --- | --- |
| `benzopyranone_ring_cleavage` (451 / 213 mols) | 0.000 · 0.3% | 0.007 · 8.3% | **0.034 · 29.5%** | 0.022 · 16.8% |
| `halogenated_quinoline_ring_cleavage` | 0.001 · 0.8% | 0.001 · 1.5% | 0.002 · 3.3% | **0.037 · 14.4%** |
| `anthraquinone_bis_co_loss` | 0.002 · 3.7% | 0.000 · 1.3% | 0.008 · 13.3% | 0.003 · 8.0% |
| `anthraquinone_ring_c4_plus_ch_loss` | 0.004 · 8.1% | 0.002 · 2.3% | 0.005 · 7.4% | 0.000 · 0.5% |
| `desoxymorphinan_aromatic_core_release` | 0.002 · 4.7% | 0.000 · 0% | 0.005 · 15.7% | 0.001 · 2.2% |
| The other six ring rules | ≤ 0.003 · ≤ 5.5% | ≤ 0.002 · ≤ 4.8% | ≤ 0.009 · ≤ 6.1% | ≤ 0.006 · ≤ 9.5% |

The template-only column sets the **chance floor of this test: mean ≤ 0.007, 0–8% of
spectra**. In MASSter these rules add a junk candidate. Their only useful effect is
*forcing* the 4-bond cut mask into the pool, which the Python backend does
(`masster_candidates.py:838-854`). The catalog's description of what they add
([`masster_smarts_rules.md`](../reference/masster_smarts_rules.md), e.g. "reassembly of
the two outer aromatic pieces") describes the byproduct. In FraGNNet both sides become DAG
nodes, so the ordinary side is the useful node there.

**New-assembly rules.** Here the template-only side is the product with the new bond.
"Admitted" marks the modes the MASSter eligibility table allows.

| Rule | Mols (+/−) | Product `[M+H]+` | Product `[M-H]-` |
| --- | --- | --- | --- |
| `aminobenzothiazole_hcn_loss` | 35 / 4 | 0.037 · 39.9% | **0.268 · 77.8%** |
| `aryl_sulfonate_so2` | 86 / 127 | 0.019 · 21.6% | 0.060 · 31.4% |
| `n_nitramine_n2o` | 10 / 5 | 0.014 · 23.5% | 0.044 · 16.4% |
| `short_haloalkyl_ammonium_c2_transfer` | 29 / 4 | 0.023 · 16.6% (admitted) | 0 |
| `aryl_sulfonamide_so2` | 446 / 239 | 0.006 · 7.9% | 0.037 · 19.2% |
| `diaryl_ketone_co_loss` | 379 / 137 | 0.010 · 3.0% | 0.036 · 13.2% |
| `methyl_benzoate_co2_transfer` | 315 / 142 | 0.010 · 9.3% (admitted) | 0.017 · 12.7% (not admitted) |
| `diarylvinyl_c2h2_loss` | 66 / 48 | 0.003 · 7.6% | 0.002 · 3.9% — **floor** |
| `n_phenylbenzamide_co_smiles` | 616 / 201 | 0.004 · 2.0% | 0.001 · 2.0% (admitted) — **floor** |

`aryl_sulfonate_so3` has no row because both of its components are ordinary, which confirms
the catalog's suspicion that it adds no assembly. `n_phenylbenzamide_co_smiles` is the
most-matched MASSter-only rule (632 molecules) and sits at the floor in the only mode it is
admitted for.

**Unregistered variants** (same method):

| Pattern | Mols (+/−) | `[M+H]+` | `[M-H]-` |
| --- | --- | --- | --- |
| Charge-tolerant sulfoxide SO loss | 113 / 41 | 0.003 · 4.0% | 0.005 · 2.4% — **floor** |
| Primary sulfonamide Ar-SO2-NH2 → SO2 loss | 178 / 67 | 0.009 · 4.7% | **0.060 · 25.4%** |
| Tertiary sulfonamide Ar-SO2-NR2 → SO2 loss | 189 / 65 | 0.001 · 0.8% | 0.012 · 5.3% |

Fixing the sulfoxide pattern is a correctness fix with no measured payoff on NIST20. The
primary sulfonamide, which the current `[NH:5]` excludes, is a better `[M-H]-` rule than the
registered secondary one.

### 2.3 Against multi-cut BFS on the same molecules

The comparison is `smpTrue` against `smpFalse_mbfs` (no SMARTS, 2 cuts per step), using
`wrecall_0.01_h4` on the molecules each rule matches. These two stats sets agree exactly on
identical DAGs (§4), so it is a same-code comparison.

- **Compound ring-cut rules.** Multi-cut BFS without any SMARTS beats `smpTrue` by 0.06–0.25:
  - morphinans 0.663 → 0.912;
  - desoxymorphinans 0.627 → 0.906;
  - anthraquinones 0.756 → 0.882;
  - xanthones 0.809 → 0.917;
  - benzopyranones 0.871 → 0.970;
  - furanocoumarins 0.858 → 0.972;
  - halogenated quinolines 0.899 → 0.956.

  The hand-mined 4-cut shortcuts recover a small part of what the enumeration they
  approximate reaches.
- **New-assembly rules.** `smpTrue` matches or beats multi-cut BFS:
  - aryl sulfonamides 0.906 vs 0.903;
  - nitramines 0.987 vs 0.962;
  - sulfonates 0.879 vs 0.886.

Corpus-wide, multi-cut BFS reaches **33.1%** of the intensity `smpTrue` misses (spectrum mean
0.920 → 0.946). The top 25 ClassyFire subclasses hold 42% of that. The largest are:

- terpene lactones, 3.4%;
- 1-benzopyrans, 2.9%;
- flavonoid glycosides, 2.8%;
- diterpenoids, 2.6%;
- indoles, 2.5%;
- benzodiazines, 2.2%;
- coumarins, 2.1%;
- O-methylated flavonoids, 1.7%;
- anthraquinones, 1.6%.

No rule targets terpenoids, steroids or chromones/flavones. 1,073 molecules carry a chromone
core, which `benzopyranone_*` never matches because it requires the 2-one.

## 3. Defects

1. **`sulfoxide_so_loss` is dead in both DAG and MASSter.** The cause is recorded in local
   notes since 2026-08-22 and still unfixed. The unit tests pass because they parse raw SMILES
   instead of `data_utils.mol_from_smiles` output. No test runs any rule on a standardized
   molecule, which is the check that would have caught this.
2. **`uniquify=True` drops valid embeddings in both anthraquinone rules**, in DAG and
   MASSter alike:
   - `anthraquinone_bis_co_loss` yields 134 partitions where every embedding gives 268;
   - `anthraquinone_ring_c4_plus_ch_loss` yields 134 where every embedding gives 536;
   - all 133 matched molecules are affected, and no other rule loses any.

   The docstring at `smarts_prepass.py:144-145` says both directions are covered; they are
   not. The remedy is `uniquify=False` plus the partition dedup the callers already do.
3. **`halogenated_quinoline_ring_cleavage` has one embedding, not two.**
   - Its docstring (`:299`) claims C6 and C7 mirrors. Quinoline has no symmetry, so only C6 is
     ever excised.
   - On a 6-haloquinoline, the excised carbon is the one bearing the halogen. That contradicts
     the rule's stated "halide-retaining" intent.
4. **Wrong comments, harmless to coverage.**
   - The NL rules' "correct formula at H_delta = −2" is wrong. The neutral formula sits at 0
     for SO2, CO and C2H2, at −1 for HCN and at +1 for N2O; all are inside ±4.
   - `_apply_smarts_prepass` says prepass pairs are depth-1 children; the single-cut path
     writes depth 2 and does not expand them. The multi-cut path writes depth 1 and does
     expand them, so the depth label means different things on the two paths.
   - The `n_nitramine_n2o` comment's R-N(NO2)-NH-R' does not match the SMARTS, which encodes
     O2N-NH-R.

## 4. Provenance traps

- **`smpFalse` stats files are from older code; the DAGs are not.**
  - `data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/*_stats_df.pkl` were written 2026-04-26.
  - On 7,492 molecules whose `smpFalse` and `smpTrue` DAGs match node-for-node,
    `wrecall_0.01_h4` still moves 0.884 → 0.900, 37.7% of spectra differ, and precision moves
    0.060 → 0.096.
  - `smpTrue` and `smpFalse_mbfs` agree exactly on their 4,118 identical DAGs.
  - A 40-molecule sample of those DAG pickles is identical key-for-key, so the drift is
    in the stats computation.

  Consequence: `smpTrue − smpFalse` read from these files (+0.013 corpus `wrecall`) is mostly
  that drift, not the prepass. The like-for-like number is the oracle script's +0.0006 cos,
  which recomputes from the DAGs.
- **`smpFalse_ringmerge` stats and `meta_info.json` are byte copies of `smpFalse`'s**, though
  its `dags/` were rebuilt 2026-07-08 and differ in all 300 sampled files. Anything that
  reads its node counts (dynamic batching) or its stats reads the base DAG's.
- **`smpTrue` probably lacks `desoxymorphinan_aromatic_core_release`.** Its DAGs were written
  2026-07-02 02:44 and the rule was committed at 14:53 the same day. No build record exists.
- NL products were missing from multi-cut prepass DAGs until `bf3820b1` (2026-09-05). This is
  recorded in `preproc_scripts/README.md`.

## 5. What would use SMARTS better, ranked

1. **Measure before porting anything to CUDA.** The 2026-09-13 review's coverage check was
   never run. On the fixed validation IDs at the production budget, measure how much of the
   7.98 feasible-but-missed points
   ([unexplained-peak split](masster/unexplained_peak_split_2026_09_22.md)) two things recover:
   - the new-assembly products;
   - the forced ring-rule masks, i.e. is each mask already inside the U-cap pool?

   §2.2 bounds the first at about 0.1% of intensity corpus-wide.
2. **If MASSter takes SMARTS at all, split the two kinds.**
   - Ring rules become **forced masks only**, with no mechanism and no byproduct component.
   - Only new-assembly rules keep an alternative partition.
   - That shrinks the CUDA port to a forced-mask list plus a handful of disconnected product
     components on about 4% of molecules. It also stops ring rules from spending mechanism
     softmax mass on a byproduct.
3. **Prune by evidence, not by literature.**
   - Drop or park `n_phenylbenzamide_co_smiles` and `diarylvinyl_c2h2_loss` (both at the floor).
   - The same goes for the four zero-support rules. Removing any of them shifts the one-based
     rule indices and `MASSTER_NUM_RULES`, which sizes the rule embedding. No SMARTS
     checkpoint exists yet, so now is the cheap time.
   - Add the primary-sulfonamide SO2 loss for `[M-H]-`.
4. **Fix the three matching defects together, behind a DAG version bump.** They are the
   sulfoxide pattern, `uniquify` and the quinoline docstring. Add one test that runs every rule
   on `mol_from_smiles` output. Changing the rules changes `smpTrue` DAG content, so it needs
   the preprocessing-reproducibility treatment in the shared rules.
5. **Mine new-assembly rules from NIST annotations instead of by hand.**
   - About 8.1% of annotated intensity has a formula the DAG cannot build
     ([unused information](unused_information_inventory_2026_09_22.md)).
   - Group those peaks by neutral-loss formula (M − annotated) and a recurring precursor
     substructure. That turns the manual `add-smarts-rule` loop into a ranked candidate list,
     and it targets exactly the kind of rule §2 shows is worth having.
   - Ring-cut residuals, including every class in §2.3, are reachable by enumeration. For
     MASSter, which has no cut-size cap, they are a budget question, not a rule question.
6. **For FraGNNet, the untested arm is curated-rules-only.** It would drop the three broad
   rules and turn `num_cut_bonds` on. The `alvhehe2` loss says nothing about it. It is
   bounded above by ring-merge and multi-cut DAGs on every ring rule's molecules, though, so
   it ranks below the MASSter items.

## 6. Method and limits

- "Explained" is a mass match with a generous H window, not an attribution. It does not
  exclude intensity another fragment already explains at the same m/z. The
  byproduct sides give its empirical chance floor.
- Molecule counts are over all 35,025 `mol_df` molecules; spectra are all NIST20
  `[M+H]+`/`[M-H]-`, not a split. The census uses the production matcher functions on the stored
  standardized `mol` objects.
- The 0.1% ceiling is a sum of per-rule means × spectra. It double-counts overlapping rules and
  includes chance matches, so it is an upper bound.
- Not checked: whether each ring rule's mask already falls inside the production U-cap pool
  (item 5.1), and ion-channel/charge handling of the products.
