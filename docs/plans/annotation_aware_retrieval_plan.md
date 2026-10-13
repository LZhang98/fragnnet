# Annotation-aware candidate retrieval

> **Status:** ACTIVE · **Updated:** 2026-10-06
> Pilot and full formula-track runs done; next is a `[M+Na]+`/no-CE policy for full-challenge numbers and a checkpoint ensemble on val.
>
> Score MassSpecGym retrieval candidates with what the predicted spectrum carries beyond its
> thresholded intensities: the full predicted support, the low-intensity tail, and per-peak
> fragment annotations. A 400-query formula-val pilot ran on `ll04i00m` (2026-10-02).
>
> - Query log-likelihood (c): +14.8 hit@1 points over cosine; sqrt-cosine recovers most of it.
> - Support (b): worse than cosine.
> - Fragment explanation cost and parsimony: add to cosine, add nothing to (c).
>
> **Full formula track done (2026-10-03),** for simulation-challenge `[M+H]+` queries only. On
> test, MASSter's likelihood scores hit@1 0.485 against 0.358 for `cos_sim_0.01`. GLACIER is
> −0.023 / −0.027 hit@1 below MASSter on `cos_sim_0.01` / `cos_sim_sqrt_0.01`, and level on
> ms-pred's entropy, the similarity its paper's retrieval numbers use. The released GLACIER
> checkpoint is the contrastively fine-tuned (CF) model. On ms-pred's own candidate table it
> reproduces the paper's w/ CF row: 49.67 / 78.20 / 90.80 vs 49.88 / 78.13 / 90.75 hit@1/5/20.
> [Full finding](../findings/masster/masster_msgym_formula_retrieval_full_2026_10_03.md)
> · [pilot finding](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md).
> Cross-model, so it sits at the top of `plans/`.

## Objective and acceptance criteria

Test whether the true molecule ranks higher among its retrieval candidates when the score uses
support, the tail below the export threshold and fragment identity, rather than cosine alone.

- **Population.** MassSpecGym 1.5 val fold, `[M+H]+` queries with a CE: 9,734 queries, the
  `ms_gym_benchmark_1_5` val split. Formula-bounded candidates first, then mass-bounded ones.
- **Gate 1 (after Step 1).** A Step 1 score, alone or blended with cosine, raises hit@1 over
  cosine. The lower bound of the 95% molecule-bootstrap interval must exceed a minimum useful gain
  that is fixed before the first measurement, and hit@5 and hit@20 must move the same way. The
  gain must survive the spectrum-blind controls below. Select the blend weight on even molecules
  and report it on odd molecules.
- **Stop rule.** If neither score (b) nor score (c) beats cosine on the formula set, cosine is not
  discarding retrieval signal that the support or tail carries. Do not build Steps 2-3.
- The test fold stays unread until the scoring method is frozen.

## Established facts

**Current scoring is cosine only.** The PubChem retrieval evaluator ranks candidates by
`cos_sim_0.01` or `cos_hun`
([eval_pubchem_retrieval_val.py:92-96](../../scripts/ms2c/eval_pubchem_retrieval_val.py#L92-L96)).
MSGYM retrieval has no scorer. The only code that reads `source_identifier` is
[01_prepare_msgym_retrieval.py](../../preproc_scripts/msgym_ms2c/01_prepare_msgym_retrieval.py).
The manifest's `peaks` column is empty on every row, so query spectra must be joined from
`data/raw/msgym_1_5/MassSpecGym1.5.tsv` by `source_identifier`. No retrieval hit rate for any of
our models is recorded in `docs/`.

**Candidate sets.** Measured 2026-09-25 from each val zip's `query_candidates.pkl.gz` in
`data/ms2c/msgym/v1_5/`, deduplicated on `(source_identifier, candidate_smiles)` as `build-proc`
does:

| | formula val | mass val |
| --- | ---: | ---: |
| all queries | 19,429 | 19,429 |
| candidate rows, all queries | 2,433,451 | 4,640,577 |
| `[M+H]+` queries with a CE | 9,734 | 9,734 |
| candidate rows, that subset | 1,609,714 | 2,417,128 |
| unique candidates, that subset | 416,967 | 587,675 |
| candidates per query, median | 222 | 253 |
| single-candidate queries | 47 | 0 |

- **The subset is the benchmark val split.** All 9,734 identifiers match
  `data/df/ms_gym_benchmark_1_5_val_dset_ids.csv`, and none is in the train ids. They cover 2,407
  distinct query SMILES.
- **The rest is out of the MSGYM model's domain.** 4,737 queries (24%) are `[M+Na]+`, and 48.7%
  have no CE (`ace` and `nce` both NaN). The MSGYM model trained only on the `[M+H]+`
  simulation-challenge set, and the 2026-09-24 scope decision in
  [project state](../project-state.md) keeps it there.
- **Score hits by 2D InChIKey.** Raw SMILES equality finds the true molecule in only 69.8% of the
  9,734 queries. The 2D InChIKey first block finds it in all of a 400-query sample, exactly once
  each. That was a sample. The scorer must assert, over all 1.61M candidate rows, that each query
  has exactly one candidate sharing the truth's first block. It must fail rather than
  silently pick one.
- **Take the truth key from MSGYM's source SMILES, not the benchmark `mol_df`** (measured
  2026-10-02). Training data is tautomer-canonicalized by `02_prepare_proc.py`; the retrieval
  build is not.
  - With the benchmark key, 660 of 9,731 val queries (95 molecules) find no truth.
  - With the source key, all 400 pilot queries find exactly one.
  - The candidates, truth included, are featurized unstandardized, unlike training data.
- **The truth is candidate 0.** MassSpecGym's JSON stores it first (3,000 of 3,000 sampled), and
  the build keeps that order. "List position" is therefore not a spectrum-blind control here,
  and any first-wins tie-break leaks the answer. The PubChem-order bias needs PubChem CIDs.
- **The retrieval `spec_df` files still say `FT`,** which the loader refuses. Relabel to `ORB`
  (MSGYM has one Orbitrap label) when building inference inputs.

**Both models export per-peak annotation.** The FraGNNet export carries a per-peak neutral
formula plus its adduct delta, `P(n|f)` over about four candidate DAG nodes per formula, and
each node's atom bitmask ([run_inference guide](../guides/run_inference.md#fragment-annotation)).
Since 2026-09-25 (committed in `4e29da7c`), MASSter fills the same columns. It builds them
from its output pathways, and the guide's MASSter paragraph lists how they differ. On 128
MassSpecGym val spectra with `ll04i00m`, every peak's formula plus adduct reproduced its m/z, and
every fragment decoded to a connected substructure of the molecule. The annotation adds about
46 KB per spectrum unfiltered and 1.8 KB at `--min_intensity 10`. For the 1.61M formula-val
candidate spectra, that makes an annotated export of about 94 GB and 4.4 GB, respectively. On MSGYM test (`ll04i00m`), the
unfiltered export has a median of 924 peaks per spectrum.

**Anchors.**
- MASSter: `ll04i00m` at epoch 60 (W&B), the checkpoint in the
  [GLACIER comparison](../findings/masster/masster_msgym_vs_glacier_released_ckpt_2026_09_24.md).
  MASSter model code has changed since that measurement (SMARTS rules, Si fix, NCE-encoder arms,
  ion-H channel work). The checkpoint's hparams set its behaviour, and the template fills keys
  added since. **Re-checked 2026-10-02:** MSGYM test mean `cos_sim_0.01` is 0.571272 against
  0.571277. The checkpoint names the retired `FT`, so it runs from a copy with `FT` renamed to
  `ORB` (weights untouched, same embedding order); see the
  [pilot finding](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#setup).
- FraGNNet: a depth-3 MSGYM config exists
  ([ms_gym_benchmark_1_5.yml](../../config/massspecgym/ms_gym_benchmark_1_5.yml)). No trained
  checkpoint or val score for it is recorded in `docs/`, and W&B has not been checked.

**Why support and the tail should carry signal.** This is evidence, not proof.
- On the NIST20 FraGNNet test export, base-peak misses are about 7:1 misranking against missing
  coverage ([dominant-peak decomposition](../findings/fragnnet/dominant_peak_outcome_decomposition.md)).
- The MASSter ranking error is a generalization failure: val ranking 0.121 against train 0.011
  ([ranking decomposition](../findings/masster/masster_ranking_decomposition_2026_09_23.md)).
- Independent runs agree less on novel molecules. Agreement falls from 0.94 to 0.75 as
  nearest-train Tanimoto drops from above 0.8 to below 0.3
  ([variance is epistemic](../findings/masster/masster_variance_novelty_2026_09_24.md)).
  Decoy candidates are unseen molecules by construction, so their intensities are the least
  reliable part of their predictions.

**Spectrum-blind baselines are high.** A structure-only model gets 26.9% top-1 under a
conventional structure-disjoint MassSpecGym split
([spurious evaluations](../notes/spurious_evaluations_spectraverse.md)). The PubChem default
ordering gets 49.98% hit@1 on the retrieval challenge
([MassSpecGym audit](../notes/massspecgym_in_the_wild_audit.md)). Every number here needs the
spectrum-blind controls next to it.

**Literature on annotation-based ranking** (web sweep, 2026-10-02; primary-source abstracts and
methods, not re-derived).
- In-silico fragmenters score each peak's best explanation independently.
  - [MetFrag 2.2](https://pmc.ncbi.nlm.nih.gov/articles/PMC4732001/) uses
    Σ RelMass^1.84 · RelInt^0.59 / Σ BDE^0.47. Fragmentation alone ranks 6% of 473 spectra
    first; retention time and reference counts supply most of its 71%.
  - [MetFrag 2.4.5](https://pmc.ncbi.nlm.nih.gov/articles/PMC6612146/) adds learned per-peak and
    per-loss terms under an explicit independence assumption.
- Forward models rank by spectral similarity only.
  - [CFM-ID 3.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC6523630/) mixes spectral similarity
    with class and reference metadata.
  - [ICEBERG](https://pmc.ncbi.nlm.nih.gov/articles/PMC12154671/),
    [FraGNNet](https://arxiv.org/abs/2404.02360), [GLACIER](https://arxiv.org/abs/2606.29161),
    [MARASON](https://arxiv.org/abs/2502.17874) and
    [FIORA](https://www.nature.com/articles/s41467-025-57422-4) put no annotation term in the
    retrieval score.
  - The nearest use of atom-level fragment consistency is
    [FRIGID](https://arxiv.org/abs/2604.16648), for generation refinement, not ranking.
- The sweep found no ranking test of structure-level parsimony and no chance correction of an
  explanation score.

## Scores

Notation: `y_k` is the query's L1-normalized intensity at m/z `m_k`. For candidate `c`, `S_c` is
its predicted m/z support at `--min_intensity -1` and `p̂_c(m)` is its predicted intensity summed
within the instrument tolerance. Candidates are predicted at the query's CE, adduct and
instrument, all of which the manifest carries.

- **(a) Cosine baseline.** `cos_sim_0.01`. MassSpecGym does not prescribe the ranking score, so
  this is our baseline choice, not a benchmark requirement; a 10 ppm or instrument-tolerance
  cosine is equally admissible. `cos_hun` is a sensitivity check.
- **(b) Chance-corrected support explainability.** `E_c = Σ_k y_k² · [m_k ∈ S_c]`, the L2 share
  of the query that `c` can explain at all. It uses the same closed form as the DAG oracle in
  [dag_oracle_scores.py](../../scripts/eval/dag_oracle_scores.py).
  - Subtract `E_c` computed on copies of the query shifted by fixed mass offsets. A denser support
    explains random peaks by chance, and the shifted copies measure that.
  - Exclude peaks within tolerance of the precursor. Every candidate in a formula-bounded set
    explains the precursor.
  - For FraGNNet, `S_c` is the DAG's formula support, which is purely structural. For MASSter it
    is the proposal-selected candidates, so part of it comes from the model. Record which one was
    used.
- **(c) Query log-likelihood under the full prediction.** `L_c = Σ_k y_k · log max(p̂_c(m_k), ε)`.
  - Suppose one candidate predicts a query peak at 0.1% of its base and another cannot form that
    peak at all. `L_c` separates them. A thresholded cosine does not.
  - Keep the precursor in this score, because the predicted precursor survival differs between
    candidates.
- **(d) Blends.** `α·z(a) + (1−α)·z(b or c)`, where `z` standardizes within each query.
- **(e) Spectrum-level competitors (added 2026-10-02).** `cos_sqrt` (0.01 Da bins, sqrt
  intensities) and unweighted spectral entropy similarity. In the pilot they recover most of
  (c)'s gain over (a). Any claim that a support, tail or annotation score adds information must
  beat these, not only linear cosine.

Break exact score ties at random, or report the average rank. Never break them by candidate list
position: the truth is candidate 0 in MassSpecGym's JSON, so a first-wins tie-break returns the
answer.

**Controls.**
- Spectrum-blind rankings: support size `|S_c|` alone and bond count alone. Heavy-atom count is
  constant on the formula track. List position is not a control here (truth at index 0); the
  PubChem-order baseline needs PubChem CIDs.
- **Consensus pseudo-query** (added 2026-10-02): every score against the top 30 bins of the mean
  predicted spectrum of the query's own candidates. It is spectrum-blind and covers every query.
- Every score computed against a different query that has the same precursor formula. This
  measures how much of each score is a candidate prior rather than a spectrum match. Only 12%
  of pilot queries have such a partner.

**Report** hit@1, hit@5 and hit@20 with 95% molecule-bootstrap intervals. Show the formula and
mass sets separately, and FT and QTOF separately. Also bin by the query molecule's nearest-train
Tanimoto, because the hypothesis predicts the gain concentrates on novel molecules.

## Plan

**Step 1: support and tail scores, no annotations.**
- Build the MSGYM scorer. It joins the query spectra from the TSV, identifies the truth by 2D
  InChIKey, applies the tie rule, and computes scores (a)-(d) plus the controls.
- Predict the formula val subset with `ll04i00m` at `--min_intensity -1`, which disables the
  filter. Only `pred_mzs` and `pred_ints` are needed.
- Size: at about 924 peaks per spectrum, 1.61M candidate spectra is about 12 GB of float32
  m/z-intensity pairs. That fits under `output/` but not in the small local `/tmp`. Scoring
  during inference avoids storing the export at all.
- Time it on 200 queries before the full run. Then run the mass set.

**Step 2: fragment-level scores (only if gate 1 passes).** *Pilot read (2026-10-02):* the
explanation cost below (as `2^−(cuts+|ΔH|)`, without the node score) and a shared-cleavage-site
parsimony variant add +3 to +5 hit@1 and +5 to +7 hit@5/20 to linear cosine. They add
−1.1 [−3.4, +1.1] hit@1 to score (c), and parsimony tracks the cost (ρ 0.92). On the formula
track this step is low priority unless Gate 1 shows (c) losing to (e). Fragment-level ideas
that the pilot did not test, and that (c) cannot already contain:
- ~~formula feasibility on the mass track~~ Tested on the 400 pilot queries (2026-10-02).
  Feasibility alone scores hit@1 0.008, a blend with `ll` picks weight 0, and peaks the true
  formula cannot explain change no `ll` ranking on either track. `ll` scores 0.828 against
  `cos` 0.623 on mass val
  ([mass-track pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#mass-track-same-400-queries-added-2026-10-02));
- substructure voting across candidates;
- a model-free structural cost, outside the model's selected support.

Original design: use the MASSter annotation export
on `ll04i00m`. Each peak's fragments come with atom bitmasks, so the bonds and ring bonds a
fragment cuts can be read off the molecule. Add two scores:
- **Explanation cost.** For each query peak, the cheapest fragment of `c` that explains it: bonds
  cut, ring bonds cut, depth and the size of the H shift. Also use the model's node score. At
  `--min_intensity -1` no peak falls below a threshold, so the node score is read for every
  fragment.
- **Score during inference.** An unfiltered annotated export of the formula val set is about
  94 GB (see above). Compute the Step 2 scores inside the inference pass, as Step 1 suggests,
  rather than storing the annotation columns.
- **Diagnostic-peak weighting.** Weight each query peak by how much the top candidates' predicted
  intensities disagree there. This removes the intensity-error noise that shared losses add
  without adding information.

**Step 3: learned reranker.** A listwise ranker over the scores from Steps 1-2, cross-fitted by
molecule on val. It must not be fitted on the train-fold retrieval sets
(`retrieval_candidates_*_train`). The true train molecules' predictions are fitted (MASSter's
train ranking gap is 0.011 against 0.121 on val), so a ranker fitted there learns to trust cosine
far more than val supports. A reranker that sees only the spectrum-blind features is its control.
Read the test fold once, after freezing.

**Out of scope for now.**
- **Substructure voting.** Query peaks vote, through every candidate's bitmasks, for the
  substructures that explain them.
- **Ambiguity sets and CE planning.** Flag candidates whose annotated spectra are
  indistinguishable, and choose the CE that best separates the top candidates. This and
  substructure voting are applications, worth building only after a score has proven itself.
- **Per-peak uncertainty as a cosine weight.** This was rejected in design. Ensemble spread is
  mostly spectrum-level: inside one spectrum, the higher-spread peak is the worse one only 58-60%
  of the time ([per-peak uncertainty](../findings/masster/masster_per_peak_uncertainty_2026_09_24.md)).
  Across candidates, unusual decoys are the uncertain ones, so down-weighting their mismatches
  would reward them. Uncertainty may enter only as a predictive spread inside score (c).

**Constraints.**
- Step 1 needs inference only, no training.
- Cluster inference follows the [submission guide](../guides/submit_cluster_runs.md).
- Queries without a CE, and `[M+Na]+` queries, need a separate CE and adduct policy, such as a CE
  grid with a max over it. They are excluded here.

**Related.** [Energy-based spectrum reranking](energy_based_spectrum_reranking_plan.md) reranks
alternative intensity allocations for one molecule. This plan ranks molecules. They share the
`--min_intensity -1` export contract.

## Current state

**Pilot complete (2026-10-02):** 400 formula-val queries, one per molecule, 72,946 candidate
spectra, on `ll04i00m`@60. Scores (a)-(d), the explanation cost, the parsimony variant, (e) and
the controls were all measured.
- hit@1: cosine 0.478, (c) 0.625 (+0.148 [+0.100, +0.195]), `cos_sqrt` 0.588, entropy 0.573,
  (b) 0.187.
- Every spectrum-blind control is 0.018–0.033 hit@1 (random 0.025).

This is not Gate 1: no minimum useful gain was fixed, and the (e), split-(c) and (c)-based blends
are post hoc. The scorer is a pilot script, `output/annotation_retrieval_pilot_2026_10_02/score_pilot.py`;
inference ran at about 150 candidate spectra/s on the local 2070 SUPER.
[Finding, numbers and artifacts](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md).

## Open questions

1. What minimum useful hit@1 gain should Gate 1 require, and over which baseline: (a) as the
   original gate states, or the best of (e)? The pilot used ε 1e-5 and shifts ±5/±7/±11 Da; Gate 1
   can keep them.
2. ~~Should candidates be standardized like training data?~~ Answered 2026-10-02: it changes
   no score beyond ±0.005 hit@1 (all intervals include zero), and it makes 10 of 400 truths tie
   with a tautomer decoy. Keep candidates as given and identify the truth by MSGYM's source key.
   [Test](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#pipeline-findings).
3. Does the goal need the no-CE and `[M+Na]+` queries, which are 50% of the val fold?

## Full formula-track run: frozen protocol (2026-10-02, before any test-fold read)

The user asked for all MassSpecGym 1.5 formula sets. This protocol is fixed before the test
fold is read.

- **Population.** Formula-track queries that both checkpoints can serve: `[M+H]+` with a CE.
  - Val: 9,731 queries, the `ms_gym_benchmark_1_5` val split, 1,609,711 candidate rows.
  - Test: 9,954 queries, exactly the simulation-challenge test set, 1,657,238 rows.
  - This is the simulation challenge's population. The `[M+Na]+` and no-CE queries in the
    retrieval zips (val 19,429, test 17,556) belong only to the molecule-retrieval challenge,
    a different task. MassSpecGym leaves the ranking score free, so every score below,
    `ll` included, is an admissible leaderboard ranking.
- **Inputs.** Candidates as given (unstandardized). Each candidate row is the query's benchmark
  row with the molecule swapped.
- **Truth.** Every candidate whose 2D key matches MSGYM's source SMILES.
  - Val: exactly one per query. Test: one for 9,935 queries, two (listed stereoisomers) for 19.
  - A query counts as a hit if any truth candidate is in the top k (MassSpecGym's 14-character
    InChIKey rule).
- **Ties.** Expected hit@k under random tie-breaking, never list position (the truth is
  candidate 0).
- **Models.** MASSter `ll04i00m`@60 (`FT`→`ORB` copy) at `--min_peak_int -1`, and GLACIER's
  released MSG checkpoint, squared, at upstream's retrieval settings (`03_run_retrieval.py` at
  `318bb7b`: `--batch-size 64 --sparse-k 100`; user decision 2026-10-03). GLACIER's output
  depends on batch composition: at batch 16 vs 64, 19% of 512 test spectra change (median 2%
  relative intensity shift, worst cosine 0.95). The first five test shards, run at batch 16 /
  sparse-k 256, are kept in `glacier_bs16_k256/` and are not used. GLACIER runs on test only and is
  scored with `cos_sim_0.01` and `cos_sim_sqrt_0.01` (user decisions, 2026-10-02/03); both models'
  `cos_sim_sqrt_0.01` on test use the eval's own CUDA binning.
- **Scores.** All four are reported:
  - (a) `cos_sim_0.01` (our baseline; MassSpecGym does not fix the ranking score), with
    `cos_hun` as a sensitivity check;
  - (c) `ll` with ε 1e-5, also without the precursor;
  - (e) `cos_sqrt`;
  - (e) `ent`;
  - `cos_hun_inst_tol` (added at the user's request before any test read), computed offline
    with the in-model code path. It reproduces the in-model value to 1e-6.
  - `cos_hun_inst_tol_sqrt`, MASSter on test only (user request, before the test read). It
    reproduces the in-model value to 8e-5.
- **Primary comparison.** (c) against (a), fixed by the pilot.
- **Spectrum-blind control.** `cos` and `ll` against the consensus pseudo-query (top 30 bins of
  the candidates' mean prediction).
- **Reporting.** hit@1, @5 and @20 with 95% molecule-bootstrap intervals (2,000 resamples),
  overall and by instrument. Val is also reported without the 400 pilot molecules.
- **Out of scope.** No blends or reranker are fitted on test; the mass track comes later.

## Next concrete check

**Superseded 2026-10-03.** The full formula track (val + test) ran under the frozen protocol
above; see the [full finding](../findings/masster/masster_msgym_formula_retrieval_full_2026_10_03.md).
Open next steps:
- a policy for `[M+Na]+` and no-CE queries, needed for leaderboard-comparable numbers;
- a query-level strongest-decoy peak comparison for the QTOF deficit;
- the mass track at full scale.

Original next check: run Gate 1 on the formula val queries outside the 400 pilot molecules
(`data/pilot/pilot_queries.csv`), with scores (a), (c) and (e) and the consensus control.
Fix the minimum useful gain and the baseline first (open question 1). Only `pred_mzs` and
`pred_ints` are needed, so the export can drop annotations or be scored inside inference. At
about 150 spectra/s the 1.61M formula-val candidates take roughly 3 GPU-hours locally. Then run
the mass set, which is where formula-level annotation evidence is still untested.

## Scoring-rule calibration: frozen before the test read (2026-10-03)

Selected on formula val only (`output/msgym_retrieval_calibration_2026_10_03/`: `calib_features.py`,
`select_val.py`, `val_selection.csv`). Val molecules were split by md5 parity of the 2D key:
120 log-likelihood variants and one- and two-term blends were ranked on half A (5,215 queries),
and the choice was reported on half B (4,516 queries).

- **Frozen rule.** `ll_g0.75_t2_e1e-4_a0.5`: Σ_k w_k log max(p̂_k, 1e-4), where w = √y / Σ√y and
  p̂_k is the mass of p^0.75 (renormalized) within 2× the instrument tolerance of query peak k.
- **Val.** Half B: hit@1 0.5689 against 0.5248 for the frozen `ll`, +0.0441 [+0.0304, +0.0581].
  All val: 0.5529 against 0.5105, +0.0424 [+0.0337, +0.0512].
- **Blends.** Reverse likelihood, predicted mass explained, cosine-sqrt and entropy each add
  +0.007–0.008 on A but −0.0003 to +0.002 on B. They are not used.
- **Knobs on A** (best over the others): γ 0.75 > 0.5 > 1.0 > 1.5; ε 1e-4 > 1e-3 > 1e-5;
  query weight √y > uniform > linear y; tolerance 2× ≈ 1× (+0.002).
- **Test read.** One read of this rule and the frozen `ll` on all 9,954 test queries,
  reported with paired molecule-bootstrap intervals. No other rule is read on test.
